"""
Baseline embedding script for the history RAG corpus.

This script:
1. Loads baseline chunks from JSON
2. Embeds each chunk using OpenAI text-embedding-3-small
3. Saves chunks with their embedding vectors and experiment metadata

Why this version is better:
- batched embedding requests
- retry logic with exponential backoff
- optional explicit dimensions parameter
- safe handling of empty input
- metadata for reproducibility
"""

import json
import os
from dotenv import load_dotenv
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIConnectionError, APIError, RateLimitError


# =========================
# Configuration
# =========================

INPUT_PATH = "data/chunks/chunks_baseline.json"
OUTPUT_PATH = "data/embeddings/embeddings_baseline_large.json"

EMBEDDING_MODEL = "text-embedding-3-large"

# text-embedding-3-small defaults to 1536 dimensions.
# Leave as None to use the model default.
EMBEDDING_DIMENSIONS: Optional[int] = None

BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.5

# Retry settings
MAX_RETRIES = 6
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 30.0


# =========================
# Helpers
# =========================

def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of chunks in {path}, got {type(data).__name__}")

    return data


def validate_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Validate that each chunk has a non-empty 'text' field.
    Raises ValueError if a problem is found.
    """
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at index {i} is not a dict.")
        if "text" not in chunk:
            raise ValueError(f"Chunk at index {i} does not contain a 'text' field.")
        if not isinstance(chunk["text"], str):
            raise ValueError(f"Chunk at index {i} has non-string 'text'.")
        if not chunk["text"].strip():
            raise ValueError(f"Chunk at index {i} has empty 'text'.")


def build_embedding_request_kwargs(texts: List[str]) -> Dict[str, Any]:
    """Build kwargs for the embeddings API call."""
    kwargs: Dict[str, Any] = {
        "input": texts,
        "model": EMBEDDING_MODEL,
    }

    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS

    return kwargs


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Send a batch of texts to the OpenAI embeddings API.
    Returns embedding vectors in the same order as input texts.
    """
    kwargs = build_embedding_request_kwargs(texts)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(**kwargs)

            # Sort by index to preserve input order
            sorted_data = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in sorted_data]

        except (RateLimitError, APIConnectionError, APIError) as e:
            is_last_attempt = attempt == MAX_RETRIES - 1
            if is_last_attempt:
                raise RuntimeError(
                    f"Embedding batch failed after {MAX_RETRIES} attempts."
                ) from e

            delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
            print(
                f"Batch failed with {type(e).__name__}. "
                f"Retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(delay)

    raise RuntimeError("Unexpected retry loop exit in embed_batch.")


def embed_chunks(chunks: List[Dict[str, Any]], client: OpenAI) -> List[Dict[str, Any]]:
    """
    Embed all chunks in batches.
    Adds an 'embedding' field to each chunk dict.
    """
    total = len(chunks)
    embedded_chunks: List[Dict[str, Any]] = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        print(f"Embedding chunks {batch_start + 1}–{batch_start + len(batch)} / {total}...")

        vectors = embed_batch(client, texts)

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Mismatch between returned vectors ({len(vectors)}) "
                f"and batch size ({len(batch)})."
            )

        for chunk, vector in zip(batch, vectors):
            embedded_chunks.append({
                **chunk,
                "embedding": vector,
            })

        if batch_start + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return embedded_chunks


def build_output_payload(embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap embedded chunks with metadata for reproducibility."""
    vector_dimension = 0
    if embedded_chunks and "embedding" in embedded_chunks[0]:
        vector_dimension = len(embedded_chunks[0]["embedding"])

    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": INPUT_PATH,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions_requested": EMBEDDING_DIMENSIONS,
            "embedding_dimensions_actual": vector_dimension,
            "batch_size": BATCH_SIZE,
            "sleep_between_batches_sec": SLEEP_BETWEEN_BATCHES,
            "total_chunks": len(embedded_chunks),
        },
        "data": embedded_chunks,
    }
    return payload


def save_output(path: str, payload: Dict[str, Any]) -> None:
    """Save payload to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAI(api_key=api_key)

    print(f"Loading chunks from {INPUT_PATH}...")
    chunks = load_chunks(INPUT_PATH)

    if not chunks:
        raise ValueError("Input chunk file is empty.")

    validate_chunks(chunks)

    print(f"Total chunks to embed: {len(chunks)}")
    print(f"Embedding model      : {EMBEDDING_MODEL}")
    print(f"Requested dimensions : {EMBEDDING_DIMENSIONS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print("=" * 60)

    embedded_chunks = embed_chunks(chunks, client)
    payload = build_output_payload(embedded_chunks)
    save_output(OUTPUT_PATH, payload)

    actual_dim = payload["metadata"]["embedding_dimensions_actual"]

    print("=" * 60)
    print("Embedding complete.")
    print(f"Total embedded chunks : {len(embedded_chunks)}")
    print(f"Output path           : {OUTPUT_PATH}")
    print(f"Vector dimension      : {actual_dim}")


if __name__ == "__main__":
    main()