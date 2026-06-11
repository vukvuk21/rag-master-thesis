"""
Baseline generator script for the history RAG corpus.

This script:
1. Accepts a user query
2. Retrieves top-k relevant chunks from ChromaDB
3. Builds a prompt with the retrieved context
4. Generates an answer using OpenAI GPT model
5. Returns the final answer with source metadata
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
COLLECTION_NAME = "history_baseline3"
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATOR_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5


# =========================
# Retrieval
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


# =========================
# Prompt building
# =========================

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a RAG prompt by combining the retrieved chunks as context
    and appending the user query.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk["metadata"].get("title", "Unknown")
        context_parts.append(f"[{i}] {title}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = (
    "You are a knowledgeable history assistant. "
    "Answer the question using only the provided context. "
    "Do not use any external knowledge. "
    "If the context does not contain enough information to answer the question, "
    "say: 'I don't have enough information in the provided context.' "
    "Be concise and include all relevant information from the context.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n\n"
    "Answer:"
    )

    return prompt


# =========================
# Generation
# =========================

def generate_answer(prompt: str, client: OpenAI) -> str:
    """Send the prompt to the OpenAI chat completion API and return the answer."""
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# =========================
# Full RAG pipeline
# =========================

def rag(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve relevant chunks and generate an answer.
    Returns the answer along with the retrieved chunks for inspection.
    """
    chunks = retrieve(query, collection, openai_client, top_k)
    prompt = build_prompt(query, chunks)
    answer = generate_answer(prompt, openai_client)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks,
    }


def print_result(result: Dict[str, Any]) -> None:
    """Print the RAG result in a readable format."""
    print(f"\nQuery: {result['query']}")
    print("=" * 60)
    print(f"Answer:\n{result['answer']}")
    print("\nSources:")
    for chunk in result["retrieved_chunks"]:
        print(f"  - {chunk['metadata']['title']} "
              f"| {chunk['metadata']['source_category']} "
              f"| distance: {chunk['distance']:.4f}")
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

    test_queries = [
        "What was the name of the Roman general who defeated Caesar at the Battle of Carrhae?"
    ]

    for query in test_queries:
        result = rag(
            query=query,
            collection=collection,
            openai_client=openai_client,
            top_k=DEFAULT_TOP_K,
        )
        print_result(result)


if __name__ == "__main__":
    main()