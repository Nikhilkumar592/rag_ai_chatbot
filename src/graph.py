"""LangGraph RAG workflow with a groundedness-grading loop.

    START -> retrieve --(relevant?)--> generate -> grade --(grounded?)--> END
                 |                        ^           |
                 +--(no)--> refuse -> END +-- retry --+ (max attempts, then refuse)
"""
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src import config


class AgentState(TypedDict, total=False):
    question: str
    context: List[str]          # chunk texts (prefixed with page number)
    retrieval_scores: List[float]
    retrieval_score: float      # mean cosine similarity of the top-k chunks
    top_score: float            # best cosine similarity
    retrieval_error: str | None # clear cause when the index is empty or unusable
    answer: str
    grounding_score: float      # 0-1 from the LLM grader
    score: float                # final confidence score
    attempts: int
    is_refusal: bool


class GroundingGrade(BaseModel):
    """Structured output of the hallucination grader."""

    grounding_score: float = Field(
        ge=0.0, le=1.0,
        description="Fraction (0-1) of the answer's claims that are directly supported by the context.",
    )
    reasoning: str = Field(description="One-sentence justification.")


GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict question-answering assistant for the 'Agentic AI' eBook.\n"
     "Answer the question using ONLY the context below. Do not use outside knowledge.\n"
     "If the context does not contain enough information, reply with exactly: "
     f"'{config.REFUSAL_MESSAGE}'\n"
     "{extra_rule}"
     "Be concise and factual.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Given CONTEXT and an ANSWER, rate how much of the "
     "answer is directly supported by the context. 1.0 = every claim is supported; "
     "0.0 = the answer is unsupported or contradicts the context. Penalise any claim "
     "that relies on outside knowledge."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])

STRICT_RETRY_RULE = (
    "Your previous answer contained claims not supported by the context. "
    "This time include ONLY statements that appear in the context.\n"
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def build_rag_graph():
    config.require_keys()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )
    vectorstore = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0,
    )
    grader = llm.with_structured_output(GroundingGrade)

    # ---------- Nodes ----------
    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        context = [f"[Page {d.metadata.get('page_number', '?')}] {d.page_content}" for d, _ in results]
        scores = [float(s) for _, s in results]
        retrieval_error = None
        if not results:
            retrieval_error = (
                "No vectors were found in the Pinecone index. "
                "Run 'python -m src.ingestion' to load the PDF embeddings, or use "
                "'python -m src.ingestion --reset' to rebuild the index."
            )
        return {
            "context": context,
            "retrieval_scores": scores,
            "retrieval_score": sum(scores) / len(scores) if scores else 0.0,
            "top_score": max(scores) if scores else 0.0,
            "retrieval_error": retrieval_error,
            "attempts": 0,
        }

    def generate_node(state: AgentState) -> Dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        msg = GENERATE_PROMPT.format_messages(
            context="\n\n".join(state["context"]),
            question=state["question"],
            extra_rule=STRICT_RETRY_RULE if attempts > 1 else "",
        )
        return {"answer": llm.invoke(msg).content.strip(), "attempts": attempts}

    def grade_node(state: AgentState) -> Dict[str, Any]:
        # The model itself declined -> honour it, no need to grade.
        if config.REFUSAL_MESSAGE.lower() in state["answer"].lower():
            return {"is_refusal": True, "answer": config.REFUSAL_MESSAGE,
                    "grounding_score": 0.0, "score": 0.0}
        grade: GroundingGrade = grader.invoke(
            GRADER_PROMPT.format_messages(
                context="\n\n".join(state["context"]), answer=state["answer"]
            )
        )
        # Confidence = 40% retrieval strength + 60% groundedness.
        # Cosine similarity for relevant chunks typically lands in ~0.3-0.7,
        # so rescale that band to 0-1 before blending.
        retrieval_conf = _clamp((state["retrieval_score"] - 0.2) / 0.4)
        confidence = 0.4 * retrieval_conf + 0.6 * grade.grounding_score
        return {"grounding_score": grade.grounding_score, "score": round(_clamp(confidence), 2),
                "is_refusal": False}

    def refuse_node(state: AgentState) -> Dict[str, Any]:
        retrieval_error = state.get("retrieval_error")
        answer = config.REFUSAL_MESSAGE
        if retrieval_error:
            answer = f"{config.REFUSAL_MESSAGE} {retrieval_error}"
        return {"answer": answer, "is_refusal": True,
                "grounding_score": 0.0, "score": 0.0,
                "retrieval_error": retrieval_error}

    # ---------- Routers ----------
    def route_after_retrieve(state: AgentState) -> str:
        if not state.get("context"):
            return "refuse"
        return "generate" if state.get("top_score", 0.0) >= config.RELEVANCE_THRESHOLD else "refuse"

    def route_after_grade(state: AgentState) -> str:
        if state.get("is_refusal") or state["grounding_score"] >= config.GROUNDING_THRESHOLD:
            return "end"
        return "retry" if state["attempts"] < config.MAX_GENERATION_ATTEMPTS else "refuse"

    # ---------- Graph ----------
    wf = StateGraph(AgentState)
    wf.add_node("retrieve", retrieve_node)
    wf.add_node("generate", generate_node)
    wf.add_node("grade", grade_node)
    wf.add_node("refuse", refuse_node)

    wf.add_edge(START, "retrieve")
    wf.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"})
    wf.add_edge("generate", "grade")
    wf.add_conditional_edges("grade", route_after_grade, {"end": END, "retry": "generate", "refuse": "refuse"})
    wf.add_edge("refuse", END)
    return wf.compile()


_graph = None


def ask(question: str) -> Dict[str, Any]:
    """Run the graph and return the API payload."""
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    result = _graph.invoke({"question": question})
    return {
        "query": question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result.get("context", []),
        "confidence_score": result.get("score", 0.0),
        "diagnostic": result.get("retrieval_error"),
    }
