"""Ingestion pipeline: download PDF -> load -> chunk -> embed -> upsert to Pinecone.

Usage:
    python -m src.ingestion            # ingest (idempotent, deterministic IDs)
    python -m src.ingestion --reset    # drop and rebuild the index first
"""
import argparse
import math
import re
import time
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.gemini_errors import is_gemini_quota_error


INGEST_BATCH_SIZE = 20
MAX_QUOTA_RETRIES = 3
DEFAULT_QUOTA_RETRY_SECONDS = 60


def _quota_retry_delay(exc: Exception) -> int:
    match = re.search(r"retry in\s+([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if match:
        return max(1, math.ceil(float(match.group(1))))
    return DEFAULT_QUOTA_RETRY_SECONDS


def _upsert_batch(
    store: PineconeVectorStore,
    documents: List[Document],
) -> None:
    texts = [document.page_content for document in documents]
    metadatas = [dict(document.metadata) for document in documents]
    ids = [f"chunk-{document.metadata['chunk_id']}" for document in documents]

    for attempt in range(MAX_QUOTA_RETRIES + 1):
        try:
            store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                batch_size=INGEST_BATCH_SIZE,
                embedding_chunk_size=len(documents),
                async_req=False,
            )
            return
        except Exception as exc:
            if not is_gemini_quota_error(exc):
                raise
            if attempt == MAX_QUOTA_RETRIES:
                raise RuntimeError(
                    "Gemini embedding quota is still exhausted after retries. "
                    "Completed batches are saved; wait for the quota to reset or "
                    "enable billing, then rerun 'python -m src.ingestion' without "
                    "--reset to continue safely."
                ) from exc
            delay = _quota_retry_delay(exc)
            print(
                f"[ingest] Gemini quota reached; retrying this batch in "
                f"{delay} seconds ({attempt + 1}/{MAX_QUOTA_RETRIES})"
            )
            time.sleep(delay)


def download_pdf(dest: Path = config.PDF_PATH) -> Path:
    """Download the eBook from Google Drive if it is not already on disk."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ingest] PDF already present: {dest}")
        return dest
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={config.GDRIVE_FILE_ID}"
    print(f"[ingest] Downloading PDF -> {dest}")
    out = gdown.download(url, str(dest), quiet=False)
    if not out or not dest.exists():
        raise RuntimeError(
            "Could not download the PDF automatically. Download it manually from "
            "the Drive link and save it as data/Ebook-Agentic-AI.pdf"
        )
    return dest


def load_and_split(pdf_path: Path) -> List[Document]:
    """Load PDF pages and split into overlapping chunks with page metadata."""
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    cleaned: List[Document] = []
    for chunk in chunks:
        text = " ".join(chunk.page_content.split())  # normalise whitespace
        if len(text) < 40:  # drop near-empty chunks (page numbers, blanks)
            continue
        chunk.page_content = text
        chunk.metadata = {
            "source": Path(pdf_path).name,
            "page_number": int(chunk.metadata.get("page", 0)) + 1,  # 1-indexed
            "chunk_id": len(cleaned),
        }
        cleaned.append(chunk)
    print(f"[ingest] {len(pages)} pages -> {len(cleaned)} chunks")
    return cleaned


def ensure_index(pc: Pinecone, name: str, reset: bool = False) -> None:
    """Create the Gemini serverless index (3072-dim, cosine) if needed."""
    existing = pc.list_indexes().names()
    if name in existing and reset:
        print(f"[ingest] Deleting existing index '{name}'")
        pc.delete_index(name)
        existing = [n for n in existing if n != name]
    elif name in existing:
        dimension = pc.describe_index(name).dimension
        if dimension != config.EMBEDDING_DIM:
            raise ValueError(
                f"Pinecone index '{name}' has dimension {dimension}, but "
                f"{config.EMBEDDING_MODEL} uses {config.EMBEDDING_DIM}. "
                "Set PINECONE_INDEX_NAME in .env to a new index name such as "
                "'agentic-ai-gemini'; do not delete the old index."
            )
    if name not in existing:
        print(f"[ingest] Creating index '{name}'")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)


def run_ingestion(reset: bool = False) -> PineconeVectorStore:
    config.require_keys()
    pdf_path = download_pdf()
    chunks = load_and_split(pdf_path)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, config.PINECONE_INDEX_NAME, reset=reset)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )
    store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    for start in range(0, len(chunks), INGEST_BATCH_SIZE):
        batch = chunks[start : start + INGEST_BATCH_SIZE]
        _upsert_batch(store, batch)
        completed = min(start + len(batch), len(chunks))
        print(f"[ingest] Upserted {completed}/{len(chunks)} vectors")
    print(f"[ingest] Upserted {len(chunks)} vectors into '{config.PINECONE_INDEX_NAME}'")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the index")
    run_ingestion(reset=parser.parse_args().reset)
