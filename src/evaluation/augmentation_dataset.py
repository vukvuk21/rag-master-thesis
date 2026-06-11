"""
Evaluation dataset augmentation script.

This script:
1. Loads existing enriched evaluation dataset
2. Tracks already used chunk IDs to avoid duplicates
3. Generates additional multi-chunk and complex questions
4. Enriches new questions with chunk texts
5. Appends to existing dataset and saves
"""

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from generate_dataset import (
    CHUNKS_PATH,
    COMPLEX_PROMPT,
    MULTI_CHUNK_PROMPT,
    GENERATOR_MODEL,
    RANDOM_SEED,
    generate_question,
    make_question_id,
    group_chunks_by_article,
    sample_stratified_exclude,
    load_chunks,
)


# =========================
# Configuration
# =========================

EXISTING_DATASET_PATH = "data/evaluation/eval_dataset_enriched.json"
OUTPUT_PATH = "data/evaluation/eval_dataset_enriched.json"

N_ADDITIONAL_MULTI = 0
N_ADDITIONAL_COMPLEX = 10


# =========================
# Sampling
# =========================

def sample_multi_chunk_pairs_stratified_fixed(
    groups: Dict[str, List[Dict]],
    chunks: List[Dict],
    n_pairs: int,
    seed: int,
    exclude_ids: set,
) -> List[tuple]:
    """
    Sample n pairs of chunks from the same article, stratified by category.
    Minimum 2 pairs per category to avoid bias.
    Excludes already used chunk IDs.
    """
    random.seed(seed)

    available_chunks = [c for c in chunks if c["chunk_id"] not in exclude_ids]

    by_category = defaultdict(list)
    for chunk in available_chunks:
        doc_id = chunk["doc_id"]
        if len(groups[doc_id]) >= 2:
            by_category[chunk["source_category"]].append(doc_id)

    by_category = {cat: list(set(ids)) for cat, ids in by_category.items()}
    n_categories = len(by_category)

    min_per_cat = 2
    base_allocated = min_per_cat * n_categories
    remainder = max(0, n_pairs - base_allocated)
    extra_per_cat = remainder // n_categories

    selected_doc_ids = []
    for cat, doc_ids in by_category.items():
        k = min(min_per_cat + extra_per_cat, len(doc_ids))
        selected_doc_ids.extend(random.sample(doc_ids, k))

    random.shuffle(selected_doc_ids)
    selected_doc_ids = selected_doc_ids[:n_pairs]

    pairs = []
    for doc_id in selected_doc_ids:
        pair = random.sample(groups[doc_id], 2)
        pairs.append((pair[0], pair[1]))

    return pairs


# =========================
# Helpers
# =========================

def load_existing_dataset(path: str) -> List[Dict]:
    """Load existing enriched evaluation dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_used_chunk_ids(dataset: List[Dict]) -> set:
    """Collect all chunk IDs already used in existing dataset."""
    used = set()
    for q in dataset:
        used.update(q.get("relevant_chunk_ids", []))
    return used


def get_last_index(dataset: List[Dict], prefix: str) -> int:
    """Get the last used index for a given question type prefix."""
    indices = []
    for q in dataset:
        qid = q.get("id", "")
        if qid.startswith(f"q_{prefix}_"):
            try:
                indices.append(int(qid.split("_")[-1]))
            except ValueError:
                pass
    return max(indices) if indices else 0


def enrich_questions(
    questions: List[Dict],
    chunk_map: Dict[str, str],
) -> List[Dict]:
    """Add relevant_chunks_text to questions."""
    for q in questions:
        q["relevant_chunks_text"] = [
            chunk_map.get(cid, "CHUNK NOT FOUND")
            for cid in q["relevant_chunk_ids"]
        ]
    return questions


# =========================
# Generators
# =========================

def generate_additional_complex(
    client: OpenAI,
    chunks: List[Dict],
    exclude_ids: set,
    n: int,
    start_index: int,
) -> List[Dict[str, Any]]:
    """Generate additional complex questions from unused chunks."""
    sampled = sample_stratified_exclude(chunks, n, RANDOM_SEED + 99, exclude_ids)

    results = []
    for i, chunk in enumerate(sampled):
        prompt = COMPLEX_PROMPT.format(
            chunk_text=chunk["text"],
            title=chunk["title"],
        )
        generated = generate_question(client, prompt)
        if not generated:
            print(f"  [complex+] Failed for chunk {chunk['chunk_id']}")
            continue

        results.append({
            "id": make_question_id("complex", start_index + i + 1),
            "type": "complex",
            "question": generated.get("question", ""),
            "reference_answer": generated.get("reference_answer", ""),
            "key_facts": generated.get("key_facts", []),
            "relevant_chunk_ids": [chunk["chunk_id"]],
            "source_titles": [chunk["title"]],
            "source_categories": [chunk["source_category"]],
        })
        print(f"  [complex+] {generated.get('question', '')[:80]}...")

    return results


def generate_additional_multi(
    client: OpenAI,
    chunks: List[Dict],
    groups: Dict[str, List[Dict]],
    exclude_ids: set,
    n: int,
    start_index: int,
) -> List[Dict[str, Any]]:
    """Generate additional multi-chunk questions from unused chunk pairs."""
    pairs = sample_multi_chunk_pairs_stratified_fixed(
        groups, chunks, n, RANDOM_SEED + 999, exclude_ids
    )

    results = []
    for i, (chunk1, chunk2) in enumerate(pairs):
        prompt = MULTI_CHUNK_PROMPT.format(
            chunk_text1=chunk1["text"],
            title1=chunk1["title"],
            index1=chunk1["chunk_index"],
            chunk_text2=chunk2["text"],
            title2=chunk2["title"],
            index2=chunk2["chunk_index"],
        )
        generated = generate_question(client, prompt)
        if not generated:
            print(f"  [multi+] Failed for pair {i + 1}")
            continue

        results.append({
            "id": make_question_id("multi", start_index + i + 1),
            "type": "multi_chunk",
            "question": generated.get("question", ""),
            "reference_answer": generated.get("reference_answer", ""),
            "key_facts": generated.get("key_facts", []),
            "relevant_chunk_ids": [chunk1["chunk_id"], chunk2["chunk_id"]],
            "source_titles": [chunk1["title"], chunk2["title"]],
            "source_categories": [chunk1["source_category"], chunk2["source_category"]],
        })
        print(f"  [multi+] {generated.get('question', '')[:80]}...")

    return results


# =========================
# Main
# =========================

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    print(f"Loading existing dataset from {EXISTING_DATASET_PATH}...")
    dataset = load_existing_dataset(EXISTING_DATASET_PATH)
    print(f"Existing questions: {len(dataset)}")

    by_type = defaultdict(int)
    for q in dataset:
        by_type[q["type"]] += 1
    print(f"  single_chunk : {by_type['single_chunk']}")
    print(f"  multi_chunk  : {by_type['multi_chunk']}")
    print(f"  complex      : {by_type['complex']}")

    print(f"\nLoading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    groups = group_chunks_by_article(chunks)
    chunk_map = {c["chunk_id"]: c["text"] for c in chunks}
    print(f"Total chunks: {len(chunks)}")

    used_ids = get_used_chunk_ids(dataset)
    print(f"Already used chunk IDs: {len(used_ids)}")

    last_complex = get_last_index(dataset, "complex")
    last_multi = get_last_index(dataset, "multi")

    print(f"\nGenerating {N_ADDITIONAL_MULTI} additional multi-chunk questions...")
    new_multi = generate_additional_multi(
        client, chunks, groups, used_ids, N_ADDITIONAL_MULTI, last_multi
    )

    used_ids.update(
        cid for q in new_multi for cid in q["relevant_chunk_ids"]
    )

    print(f"\nGenerating {N_ADDITIONAL_COMPLEX} additional complex questions...")
    new_complex = generate_additional_complex(
        client, chunks, used_ids, N_ADDITIONAL_COMPLEX, last_complex
    )

    # Enrich new questions with chunk texts
    new_multi = enrich_questions(new_multi, chunk_map)
    new_complex = enrich_questions(new_complex, chunk_map)

    dataset.extend(new_multi)
    dataset.extend(new_complex)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    by_type_final = defaultdict(int)
    for q in dataset:
        by_type_final[q["type"]] += 1

    print("\n" + "=" * 60)
    print("Augmentation complete.")
    print(f"Added multi questions   : {len(new_multi)}")
    print(f"Added complex questions : {len(new_complex)}")
    print(f"\nFinal dataset:")
    print(f"  single_chunk : {by_type_final['single_chunk']}")
    print(f"  multi_chunk  : {by_type_final['multi_chunk']}")
    print(f"  complex      : {by_type_final['complex']}")
    print(f"  Total        : {len(dataset)}")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()