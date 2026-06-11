"""
Vector store ingestion script for the baseline RAG configuration.

This script:
1. Loads embedded chunks from JSON
2. Creates a persistent ChromaDB collection
3. Inserts all chunks with their embeddings and metadata
"""

import json
import os
import sys
from typing import Any, Dict, List

import chromadb


# =========================
# Configuration
# =========================

INPUT_PATH = "data/embeddings/embeddings_baseline.json"
CHROMA_PATH = "data/chroma/baseline"
COLLECTION_NAME = "history_baseline"


# =========================
# Helpers
# =========================

def load_embedded_chunks(path: str) -> List[Dict[str, Any]]:
    """Load embedded chunks from the JSON output of the embedder script."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Handle both raw list and wrapped payload formats
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    raise ValueError(f"Unexpected format in {path}")


def build_chroma_batch(chunks: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Extract ids, embeddings, documents, and metadata from chunk dicts
    into separate lists for ChromaDB ingestion.
    """
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "doc_id": chunk.get("doc_id", ""),
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "source_category": chunk.get("source_category", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "token_count": chunk.get("token_count", 0),
        })

    return {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }


def ingest_to_chroma(
    chunks: List[Dict[str, Any]],
    chroma_path: str,
    collection_name: str,
    batch_size: int = 500,
) -> None:
    """
    Insert all embedded chunks into a persistent ChromaDB collection.
    Uses cosine similarity as the distance metric.
    """
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if it exists to ensure clean state
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection: {collection_name}")

    total = len(chunks)
    inserted = 0

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        chroma_batch = build_chroma_batch(batch)

        collection.add(
            ids=chroma_batch["ids"],
            embeddings=chroma_batch["embeddings"],
            documents=chroma_batch["documents"],
            metadatas=chroma_batch["metadatas"],
        )

        inserted += len(batch)
        print(f"Inserted {inserted} / {total} chunks...")

    print(f"Ingestion complete. Total inserted: {inserted}")


# =========================
# Main
# =========================

def main() -> None:
    print(f"Loading embedded chunks from {INPUT_PATH}...")
    chunks = load_embedded_chunks(INPUT_PATH)
    print(f"Loaded {len(chunks)} chunks")
    print("=" * 60)

    ingest_to_chroma(
        chunks=chunks,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("=" * 60)
    print(f"ChromaDB saved to: {CHROMA_PATH}")
    print(f"Collection name  : {COLLECTION_NAME}")


if __name__ == "__main__":
    main()