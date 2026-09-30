"""FastAPI interface.

Run with either:
    uvicorn app:app --reload
    python app.py
"""
import os
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import graph as rag
from src.gemini_errors import explain_gemini_error, is_gemini_quota_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag._graph = rag.build_rag_graph()  # fail fast on missing keys / index
    yield


app = FastAPI(title="Agentic AI RAG API", version="1.0.0", lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What is Agentic AI?"])


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float
    diagnostic: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=422, detail="query must not be empty")
    try:
        return rag.ask(question)
    except Exception as exc:
        if is_gemini_quota_error(exc):
            raise HTTPException(status_code=503, detail=explain_gemini_error(exc)) from exc
        raise HTTPException(status_code=502, detail=f"RAG pipeline error: {exc}") from exc
