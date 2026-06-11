"""
Evaluation dataset generation script.

This script:
1. Samples chunks from the baseline chunk file (stratified by category)
2. Uses GPT to generate questions of three types:
   - single-chunk: answer contained in one chunk
   - multi-chunk: answer requires combining two chunks from the SAME article
   - complex: requires summarization or inference, grounded in explicit chunk content
3. Saves the generated dataset to JSON for manual review and filtering
"""

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Configuration
# =========================

CHUNKS_PATH = "data/chunks/chunks_baseline.json"
OUTPUT_PATH = "data/evaluation/eval_dataset_raw.json"

N_SINGLE_CHUNK = 70
N_MULTI_CHUNK = 25
N_COMPLEX = 25

GENERATOR_MODEL = "gpt-4o"
RANDOM_SEED = 42


# =========================
# Prompts
# =========================

SINGLE_CHUNK_PROMPT = """You are creating an evaluation dataset for a history RAG system.

Given the following text chunk from a history article, generate one question that:
- Can be answered ONLY using information in this chunk
- Has a specific, factual answer (not vague or opinion-based)
- Is not trivially obvious from the chunk title alone
- Cannot be answered from general knowledge without reading the chunk

Also provide:
- A reference answer based strictly on the chunk
- A list of 2-4 key facts that a good answer must contain

Respond in JSON format only, no extra text:
{{
  "question": "...",
  "reference_answer": "...",
  "key_facts": ["...", "...", "..."]
}}

Chunk:
{chunk_text}

Article title: {title}"""


MULTI_CHUNK_PROMPT = """You are creating an evaluation dataset for a history RAG system.

Given the following two text chunks from the SAME history article, generate one question that:
- Requires information from BOTH chunks to answer fully
- Involves comparison, cause-effect, or connecting two events or concepts
- Has a clear factual answer
- Cannot be answered from general knowledge alone

Also provide:
- A reference answer based strictly on both chunks
- A list of 2-4 key facts that a good answer must contain

Respond in JSON format only, no extra text:
{{
  "question": "...",
  "reference_answer": "...",
  "key_facts": ["...", "...", "..."]
}}

Chunk 1 ({title1}, chunk index {index1}):
{chunk_text1}

Chunk 2 ({title2}, chunk index {index2}):
{chunk_text2}"""


COMPLEX_PROMPT = """You are creating an evaluation dataset for a history RAG system.

Given the following text chunk from a history article, generate one question that:
- Requires summarization or inference, not just a direct fact lookup
- Has a clear answer that can be derived from explicit information in the chunk
- Tests deeper understanding of the content
- Must be answerable strictly from the chunk — do not require outside knowledge
- The answer must be grounded in explicit information from the chunk

Also provide:
- A reference answer based strictly on the chunk
- A list of 2-4 key facts that a good answer must contain

Respond in JSON format only, no extra text:
{{
  "question": "...",
  "reference_answer": "...",
  "key_facts": ["...", "...", "..."]
}}

Chunk:
{chunk_text}

Article title: {title}"""


# =========================
# Sampling
# =========================

def sample_stratified(chunks: List[Dict], n: int, seed: int) -> List[Dict]:
    """
    Sample n chunks proportionally across source categories.
    Ensures all categories are represented in the sample.
    """
    random.seed(seed)

    by_category = defaultdict(list)
    for chunk in chunks:
        by_category[chunk["source_category"]].append(chunk)

    n_categories = len(by_category)
    per_category = max(1, n // n_categories)

    sampled = []
    for category, cat_chunks in by_category.items():
        k = min(per_category, len(cat_chunks))
        sampled.extend(random.sample(cat_chunks, k))

    random.shuffle(sampled)

    if len(sampled) >= n:
        return sampled[:n]

    # Top up if needed
    already_selected = {c["chunk_id"] for c in sampled}
    remainder = [c for c in chunks if c["chunk_id"] not in already_selected]
    sampled.extend(random.sample(remainder, n - len(sampled)))
    return sampled

def sample_stratified_exclude(
    chunks: List[Dict],
    n: int,
    seed: int,
    exclude_ids: set,
) -> List[Dict]:
    """Sample n chunks stratified by category, excluding already selected chunks."""
    available = [c for c in chunks if c["chunk_id"] not in exclude_ids]
    return sample_stratified(available, n, seed)

def sample_multi_chunk_pairs_stratified(
    groups: Dict[str, List[Dict]],
    chunks: List[Dict],
    n_pairs: int,
    seed: int,
) -> List[tuple]:
    """
    Sample n pairs of chunks from the same article, stratified by category.
    Only considers articles with at least 2 chunks.
    """
    random.seed(seed)

    by_category = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if len(groups[doc_id]) >= 2:
            by_category[chunk["source_category"]].append(doc_id)

    # Deduplicate doc_ids per category
    by_category = {cat: list(set(ids)) for cat, ids in by_category.items()}

    n_categories = len(by_category)
    per_category = max(1, n_pairs // n_categories)

    selected_doc_ids = []
    for cat, doc_ids in by_category.items():
        k = min(per_category, len(doc_ids))
        selected_doc_ids.extend(random.sample(doc_ids, k))

    selected_doc_ids = selected_doc_ids[:n_pairs]

    pairs = []
    for doc_id in selected_doc_ids:
        pair = random.sample(groups[doc_id], 2)
        pairs.append((pair[0], pair[1]))

    return pairs


# =========================
# Helpers
# =========================

def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load baseline chunks from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_chunks_by_article(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by their doc_id for multi-chunk pairing."""
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk["doc_id"]].append(chunk)
    return groups


def generate_question(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """
    Send prompt to GPT and parse the JSON response.
    Returns empty dict if parsing fails.
    """
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}


def make_question_id(prefix: str, index: int) -> str:
    """Generate a padded question ID like q_single_001."""
    return f"q_{prefix}_{index:03d}"


# =========================
# Generators
# =========================

def generate_single_chunk_questions(
    client: OpenAI,
    chunks: List[Dict],
) -> List[Dict[str, Any]]:
    """Generate single-chunk questions."""
    results = []
    for i, chunk in enumerate(chunks):
        prompt = SINGLE_CHUNK_PROMPT.format(
            chunk_text=chunk["text"],
            title=chunk["title"],
        )
        generated = generate_question(client, prompt)
        if not generated:
            print(f"  [single] Failed to parse response for chunk {chunk['chunk_id']}")
            continue

        results.append({
            "id": make_question_id("single", i + 1),
            "type": "single_chunk",
            "question": generated.get("question", ""),
            "reference_answer": generated.get("reference_answer", ""),
            "key_facts": generated.get("key_facts", []),
            "relevant_chunk_ids": [chunk["chunk_id"]],
            "source_titles": [chunk["title"]],
            "source_categories": [chunk["source_category"]],
        })
        print(f"  [single] {generated.get('question', '')[:80]}...")

    return results


def generate_multi_chunk_questions(
    client: OpenAI,
    pairs: List[tuple],
) -> List[Dict[str, Any]]:
    """Generate multi-chunk questions using pairs from the same article."""
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
            print(f"  [multi] Failed to parse response for pair {i + 1}")
            continue

        results.append({
            "id": make_question_id("multi", i + 1),
            "type": "multi_chunk",
            "question": generated.get("question", ""),
            "reference_answer": generated.get("reference_answer", ""),
            "key_facts": generated.get("key_facts", []),
            "relevant_chunk_ids": [chunk1["chunk_id"], chunk2["chunk_id"]],
            "source_titles": [chunk1["title"], chunk2["title"]],
            "source_categories": [chunk1["source_category"], chunk2["source_category"]],
        })
        print(f"  [multi] {generated.get('question', '')[:80]}...")

    return results


def generate_complex_questions(
    client: OpenAI,
    chunks: List[Dict],
) -> List[Dict[str, Any]]:
    """Generate complex inference or summarization questions."""
    results = []
    for i, chunk in enumerate(chunks):
        prompt = COMPLEX_PROMPT.format(
            chunk_text=chunk["text"],
            title=chunk["title"],
        )
        generated = generate_question(client, prompt)
        if not generated:
            print(f"  [complex] Failed to parse response for chunk {chunk['chunk_id']}")
            continue

        results.append({
            "id": make_question_id("complex", i + 1),
            "type": "complex",
            "question": generated.get("question", ""),
            "reference_answer": generated.get("reference_answer", ""),
            "key_facts": generated.get("key_facts", []),
            "relevant_chunk_ids": [chunk["chunk_id"]],
            "source_titles": [chunk["title"]],
            "source_categories": [chunk["source_category"]],
        })
        print(f"  [complex] {generated.get('question', '')[:80]}...")

    return results


# =========================
# Main
# =========================

def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Total chunks available: {len(chunks)}")

    groups = group_chunks_by_article(chunks)
    print(f"Total articles: {len(groups)}")

    # Sample stratified by category
    single_chunks = sample_stratified(chunks, N_SINGLE_CHUNK, RANDOM_SEED)
    single_ids = {c["chunk_id"] for c in single_chunks}
    complex_chunks = sample_stratified_exclude(chunks, N_COMPLEX, RANDOM_SEED + 1, single_ids)
    multi_pairs = sample_multi_chunk_pairs_stratified(
        groups, chunks, N_MULTI_CHUNK, RANDOM_SEED
    )

    # Show category distribution of sampled chunks
    print("\nSingle-chunk sample distribution:")
    cat_counts = defaultdict(int)
    for c in single_chunks:
        cat_counts[c["source_category"]] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<35} {count}")

    print(f"\nGenerating {N_SINGLE_CHUNK} single-chunk questions...")
    single_questions = generate_single_chunk_questions(client, single_chunks)

    print(f"\nGenerating {len(multi_pairs)} multi-chunk questions...")
    multi_questions = generate_multi_chunk_questions(client, multi_pairs)

    print(f"\nGenerating {N_COMPLEX} complex questions...")
    complex_questions = generate_complex_questions(client, complex_chunks)

    all_questions = single_questions + multi_questions + complex_questions

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Dataset generation complete.")
    print(f"Single-chunk questions : {len(single_questions)}")
    print(f"Multi-chunk questions  : {len(multi_questions)}")
    print(f"Complex questions      : {len(complex_questions)}")
    print(f"Total                  : {len(all_questions)}")
    print(f"Output path            : {OUTPUT_PATH}")
    print("=" * 60)
    print("\nNext step: manually review and filter the dataset.")


if __name__ == "__main__":
    main()