# Agentic AI eBook - RAG Chatbot (LangGraph + Pinecone)

A Retrieval-Augmented Generation chatbot that answers questions **strictly** from the
*Agentic AI* eBook. Built in plain Python with LangGraph, Pinecone, Gemini embeddings and FastAPI
(plus an optional Streamlit UI).

## Features
- PDF ingestion with page-number metadata and idempotent Pinecone upserts
- LangGraph state machine: `retrieve -> generate -> grade` with a retry loop and a refusal path
- Hallucination grading (LLM-as-judge) plus retrieval similarity blended into a confidence score
- Out-of-scope questions are refused before the LLM is even called
- Response payload: `query`, `final_answer`, `retrieved_context_chunks`, `confidence_score`

## Architecture

```
                 +-----------+   top-k chunks    +----------+
  question ----> | retrieve  | ----------------> | generate | (LLM, context-only prompt)
                 +-----------+                   +----------+
                       |                              |
          best similarity < threshold                 v
                       |                         +---------+   score < threshold
                       v                         |  grade  | -------------------+
                 +-----------+                   +---------+                    |
                 |  refuse   | <----- attempts exhausted ---------------  retry -> generate
                 +-----------+                        | grounded
                       |                              v
                       +-----------------------------> END
```

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| `src/ingestion.py` | PyPDF, RecursiveCharacterTextSplitter | Download, parse, chunk (1000/200), embed, upsert |
| Vector DB | Pinecone serverless (3072-d, cosine) | Semantic search with page metadata |
| `src/graph.py` | LangGraph | retrieve / generate / grade / refuse nodes + routing |
| `app.py` | FastAPI | `POST /chat` endpoint |
| `streamlit_app.py` | Streamlit | Optional chat UI with context side panel |

### Confidence score
`confidence = 0.4 * retrieval_strength + 0.6 * grounding_score`
- `retrieval_strength`: mean cosine similarity of the top-k chunks, rescaled from the 0.2-0.6 band to 0-1
- `grounding_score`: 0-1 from a second LLM call that checks each claim in the answer against the chunks
- Refusals always return `0.0`

### Groundedness safeguards (3 layers)
1. **Similarity gate** - if the best chunk is below `RELEVANCE_THRESHOLD` (0.30), refuse immediately.
2. **Context-only prompt** - the LLM must answer only from context or return a fixed refusal sentence.
3. **Grader + retry** - answers scoring below `GROUNDING_THRESHOLD` (0.60) are regenerated once with a stricter prompt, then refused.

## Setup

Use Python 3.12 for this project. Some dependencies do not yet support Python 3.14.

### Windows (Command Prompt)

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace any old `OPENAI_API_KEY` entry with a Gemini key, named
`GEMINI_API_KEY`; an OpenAI key cannot be used with Gemini. Keep your
`PINECONE_API_KEY`. Create a Gemini key in
[Google AI Studio](https://aistudio.google.com/api-keys). The Gemini API has a free tier
with model and usage limits; review its
[pricing and data terms](https://ai.google.dev/gemini-api/docs/pricing) before use. Do not
put real credentials in `.env.example` or commit `.env`.

## Ingest the eBook

```bash
python -m src.ingestion           # downloads data/Ebook-Agentic-AI.pdf, chunks, embeds, upserts
python -m src.ingestion --reset   # optional: rebuild the index from scratch
```
If the automatic download fails, save the PDF manually as `data/Ebook-Agentic-AI.pdf` and re-run.

## Gemini provider and existing Pinecone index

This project uses Gemini for both chat and embeddings. If the Gemini free-tier quota is
exhausted, wait for the limit to reset or enable billing for the Google AI Studio project;
check the current [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). These
limits are separate from any Google One or other consumer subscription.

Gemini embeddings have 3072 dimensions and are not compatible with the old 1536-dimensional
OpenAI index. Set `PINECONE_INDEX_NAME=agentic-ai-gemini` in `.env` before ingesting; this
creates a separate index and leaves the old one untouched. Ingestion and chat must use the
same Gemini embedding model and index. If ingestion is interrupted, rerun
`python -m src.ingestion` without `--reset`.
Ingestion upserts deterministic-ID batches and retries temporary Gemini quota
throttling. If the quota remains exhausted, completed batches are retained; wait
for the limit to reset or enable billing, then rerun without `--reset`.

## Run

**API**
```bash
uvicorn app:app --reload
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query": "What is Agentic AI?"}'
```
Interactive docs: http://127.0.0.1:8000/docs

**Streamlit UI (optional)**
```bash
streamlit run streamlit_app.py
```

### Example response
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["[Page 3] ...", "[Page 4] ..."],
  "confidence_score": 0.9
}
```

## Test

```bash
python tests_sample_queries.py                              # direct call
python tests_sample_queries.py --api http://127.0.0.1:8000  # via the API
```
Runs 6 queries (5 in-scope + "What is the capital of France?"), writes `sample_outputs.json`,
and asserts the payload shape and that the out-of-scope query is refused.

## Configuration
All tunables live in `src/config.py` and can be overridden via `.env`:
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RELEVANCE_THRESHOLD`, `GROUNDING_THRESHOLD`, `GEMINI_MODEL`.
The default chat model is `gemini-3.8-flash`.
If good questions get refused, lower `RELEVANCE_THRESHOLD`; if off-topic ones slip through, raise it.

## Project structure
```
rag-agentic-ai/
├── data/                  # PDF is downloaded here
├── src/
│   ├── config.py          # env + constants
│   ├── ingestion.py       # PDF -> chunks -> Pinecone
│   └── graph.py           # LangGraph workflow
├── app.py                 # FastAPI
├── streamlit_app.py       # optional UI
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
└── README.md
```
