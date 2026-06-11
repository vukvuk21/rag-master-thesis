"""
Hybrid retrieval + reranking for the history RAG corpus.

Pipeline:
1. Load baseline chunks for BM25
2. Build BM25 index over chunk texts
3. Embed the query with BGE-M3
4. Retrieve dense candidates from existing Chroma BGE-M3 collection
5. Retrieve sparse candidates with BM25
6. Fuse dense + sparse rankings using Reciprocal Rank Fusion
7. Rerank fused candidates using a cross-encoder reranker
8. Return final top-k chunks in the same format expected by the evaluator
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from rank_bm25 import BM25Okapi


# =========================
# Configuration
# =========================

CHUNKS_PATH = "data/chunks/chunks_baseline.json"

DENSE_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
FUSION_CANDIDATES = 30
FINAL_TOP_K = 10

RRF_K = 60

BGE_MAX_LENGTH = 8192
RERANK_MAX_LENGTH = 512

USE_FP16 = True
RERANK_BATCH_SIZE = 8


# =========================
# Tokenization for BM25
# =========================

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercases text and keeps alphanumeric word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


# =========================
# Chunk loading
# =========================

def load_chunks(path: str = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """
    Load chunks from JSON.

    Supports two formats:
    1. Plain list of chunks
    2. Payload with {"metadata": ..., "data": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "data" in payload:
        chunks = payload["data"]
    elif isinstance(payload, list):
        chunks = payload
    else:
        raise ValueError(
            f"Unsupported chunk JSON format in {path}. "
            "Expected list or dict with 'data'."
        )

    normalized_chunks = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at index {i} is not a dict.")

        text = chunk.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk at index {i} has missing or empty text.")

        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or chunk.get("metadata", {}).get("chunk_id")
        )

        if chunk_id is None:
            raise ValueError(f"Chunk at index {i} has no chunk_id/id.")

        metadata = chunk.get("metadata", {})

        if not metadata:
            metadata = {
                "title": chunk.get("title", chunk.get("article_title", "")),
                "source_category": chunk.get("source_category", ""),
                "url": chunk.get("url", ""),
                "article_id": chunk.get("article_id", ""),
                "chunk_index": chunk.get("chunk_index", i),
            }

        normalized_chunks.append({
            "chunk_id": str(chunk_id),
            "text": text,
            "metadata": metadata,
        })

    return normalized_chunks


def build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build chunk_id -> chunk lookup."""
    return {chunk["chunk_id"]: chunk for chunk in chunks}


# =========================
# BM25
# =========================

def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """Build BM25 index over chunk texts."""
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(
    query: str,
    bm25_index: BM25Okapi,
    chunks: List[Dict[str, Any]],
    top_k: int = BM25_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks using BM25."""
    query_tokens = tokenize(query)
    scores = bm25_index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = chunks[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "bm25_score": float(scores[idx]),
            "bm25_rank": rank,
            "retrieval_source": "bm25",
        })

    return results


# =========================
# BGE-M3 dense query embedding
# =========================

def load_bge_model() -> BGEM3FlagModel:
    """Load BGE-M3 model for query embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dense model: {DENSE_MODEL_NAME}")
    print(f"Device             : {device}")
    print(f"Use FP16           : {USE_FP16 and device == 'cuda'}")

    model = BGEM3FlagModel(
        DENSE_MODEL_NAME,
        use_fp16=USE_FP16 and device == "cuda",
    )

    return model


def embed_query_bge(query: str, model: BGEM3FlagModel) -> List[float]:
    """Embed one query using BGE-M3 dense vector output."""
    output = model.encode(
        [query],
        batch_size=1,
        max_length=BGE_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vec = output["dense_vecs"][0]

    if hasattr(dense_vec, "tolist"):
        return dense_vec.tolist()

    return list(dense_vec)


def dense_retrieve(
    query: str,
    collection: Any,
    bge_model: BGEM3FlagModel,
    top_k: int = DENSE_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from existing Chroma collection
    using BGE-M3 query embedding.
    """
    query_vector = embed_query_bge(query, bge_model)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, chunk_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        retrieved.append({
            "chunk_id": str(chunk_id),
            "text": documents[i],
            "metadata": metadata,
            "distance": float(distances[i]),
            "dense_rank": i + 1,
            "retrieval_source": "dense",
        })

    return retrieved


# =========================
# Reciprocal Rank Fusion
# =========================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = RRF_K,
    fusion_k: int = FUSION_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    score(doc) = sum(1 / (rrf_k + rank))
    """
    scores = defaultdict(float)
    candidates: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]

            scores[chunk_id] += 1.0 / (rrf_k + rank)

            if chunk_id not in candidates:
                candidates[chunk_id] = dict(item)
            else:
                candidates[chunk_id].update({
                    key: value
                    for key, value in item.items()
                    if key not in candidates[chunk_id]
                })

    fused = []

    for chunk_id, score in scores.items():
        item = candidates[chunk_id]
        item["rrf_score"] = float(score)
        fused.append(item)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused[:fusion_k]


# =========================
# Reranking
# =========================

def load_reranker() -> FlagReranker:
    """Load BGE reranker model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading reranker: {RERANKER_MODEL_NAME}")
    print(f"Device          : {device}")
    print(f"Use FP16        : {USE_FP16 and device == 'cuda'}")

    reranker = FlagReranker(
        RERANKER_MODEL_NAME,
        use_fp16=USE_FP16 and device == "cuda",
    )

    return reranker


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker: FlagReranker,
    final_k: int = FINAL_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks with cross-encoder reranker.
    """
    if not candidates:
        return []

    pairs = [[query, candidate["text"]] for candidate in candidates]

    scores = reranker.compute_score(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )

    if isinstance(scores, float):
        scores = [scores]

    reranked = []

    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_results = reranked[:final_k]

    for rank, item in enumerate(final_results, start=1):
        item["final_rank"] = rank

    return final_results


# =========================
# Full hybrid + rerank retrieval
# =========================

def hybrid_rerank_retrieve(
    query: str,
    collection: Any,
    bge_model: BGEM3FlagModel,
    bm25_index: BM25Okapi,
    all_chunks: List[Dict[str, Any]],
    reranker: FlagReranker,
    final_k: int = FINAL_TOP_K,
    dense_k: int = DENSE_CANDIDATES,
    bm25_k: int = BM25_CANDIDATES,
    fusion_k: int = FUSION_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:
    dense BGE-M3 retrieval + BM25 + RRF + reranking.
    """
    dense_results = dense_retrieve(
        query=query,
        collection=collection,
        bge_model=bge_model,
        top_k=dense_k,
    )

    bm25_results = bm25_retrieve(
        query=query,
        bm25_index=bm25_index,
        chunks=all_chunks,
        top_k=bm25_k,
    )

    fused_candidates = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        rrf_k=RRF_K,
        fusion_k=fusion_k,
    )

    final_results = rerank_candidates(
        query=query,
        candidates=fused_candidates,
        reranker=reranker,
        final_k=final_k,
    )

    return final_results


# =========================
# Setup helper for evaluator
# =========================

def setup_hybrid_retriever(
    chunks_path: str = CHUNKS_PATH,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    BM25Okapi,
    BGEM3FlagModel,
    FlagReranker,
]:
    """
    Load everything needed for hybrid retrieval.

    Returns:
    - all_chunks
    - chunk_lookup
    - bm25_index
    - bge_model
    - reranker
    """
    print(f"Loading chunks for BM25 from: {chunks_path}")
    all_chunks = load_chunks(chunks_path)
    chunk_lookup = build_chunk_lookup(all_chunks)

    print(f"Loaded chunks: {len(all_chunks)}")

    print("Building BM25 index...")
    bm25_index = build_bm25_index(all_chunks)

    bge_model = load_bge_model()
    reranker = load_reranker()

    print("Hybrid retriever setup complete.")
    print("=" * 60)

    return all_chunks, chunk_lookup, bm25_index, bge_model, reranker