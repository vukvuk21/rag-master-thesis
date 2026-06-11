"""
Baseline retrieval script for the history RAG corpus.

This script:
1. Embeds a user query using the same model as the corpus
2. Queries the ChromaDB collection for the most similar chunks
3. Returns the top-k chunks with their metadata and similarity scores
"""

import os
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Configuration
# =========================

CHROMA_PATH = "data/chroma/baseline"
COLLECTION_NAME = "history_baseline"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5


# =========================
# Helpers
# =========================

def get_collection(chroma_path: str, collection_name: str) -> chromadb.Collection:
    """Load an existing ChromaDB collection from disk."""
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=collection_name)


def embed_query(query: str, client: OpenAI) -> List[float]:
    """Embed a single query string using the OpenAI embedding model."""
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def retrieve(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top-k most similar chunks from ChromaDB.
    Returns a list of result dicts with text, metadata, and distance score.
    """
    query_vector = embed_query(query, openai_client)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return retrieved


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Print retrieved chunks in a readable format."""
    print(f"\nQuery: {query}")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['metadata']['title']} "
              f"| category: {result['metadata']['source_category']} "
              f"| distance: {result['distance']:.4f}")
        print(f"    {result['text'][:300].replace(chr(10), ' ')}...")
    print("=" * 60)


# =========================
# Main
# =========================

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    openai_client = OpenAI(api_key=api_key)
    collection = get_collection(CHROMA_PATH, COLLECTION_NAME)

    print(f"Collection loaded: {COLLECTION_NAME}")
    print(f"Total chunks in collection: {collection.count()}")

    # Test queries
    test_queries = [
        "Who was Julius Caesar?",
        "What caused World War I?",
        "How did the Roman Empire fall?",
    ]

    for query in test_queries:
        results = retrieve(
            query=query,
            collection=collection,
            openai_client=openai_client,
            top_k=DEFAULT_TOP_K,
        )
        print_results(query, results)


if __name__ == "__main__":
    main()