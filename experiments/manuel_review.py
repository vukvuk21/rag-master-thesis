import json
import os
import random

RESULTS_PATH = "experiments/results/results_hybrid_bge_m3_bm25_rerank_k10.json"
DATASET_PATH = "data/evaluation/eval_dataset_enriched.json"
CHUNK_STORE_PATH = "data/chunks/chunks_baseline.json"
OUTPUT_PATH = "experiments/results/manual_hybrid.json"

SAMPLES_PER_TYPE = 15
RANDOM_SEED = 42

# None = print FULL chunk text
CHUNK_PREVIEW_LEN = None

ERROR_TYPES = [
    "NONE",
    "HALLUCINATION",
    "MISATTRIBUTION",
    "OMISSION",
    "RETRIEVAL_FAIL",
]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk_store(path: str) -> dict:
    chunks = load_json(path)

    # Supports both:
    # 1. plain list of chunks
    # 2. {"metadata": ..., "data": [...]}
    if isinstance(chunks, dict) and "data" in chunks:
        chunks = chunks["data"]

    return {c["chunk_id"]: c["text"] for c in chunks}


def sample_questions(results: list, n_per_type: int, seed: int) -> list:
    random.seed(seed)

    by_type = {}

    for r in results:
        qtype = r["type"]

        if qtype not in by_type:
            by_type[qtype] = []

        by_type[qtype].append(r)

    sampled = []

    for qtype, items in by_type.items():
        k = min(n_per_type, len(items))
        sampled.extend(random.sample(items, k))

    return sampled


def get_chunk_texts(q: dict, chunk_store: dict) -> list:
    return [
        chunk_store.get(cid, f"[chunk not found: {cid}]")
        for cid in q.get("retrieved_chunk_ids", [])
    ]


def get_score(prompt: str, min_val: float = 1.0, max_val: float = 5.0) -> float:
    while True:
        try:
            raw = input(prompt).strip().replace(",", ".")
            val = float(raw)

            if min_val <= val <= max_val:
                return val

            print(f"Please enter a number between {min_val} and {max_val}.")

        except ValueError:
            print("Invalid input. Use a number like 4, 4.5, or 4.7.")


def get_error_type() -> str:
    print("\nERROR TYPES:")

    for i, e in enumerate(ERROR_TYPES):
        print(f"  {i} — {e}")

    while True:
        try:
            idx = int(input("\nSelect error type (0-4): "))

            if 0 <= idx < len(ERROR_TYPES):
                return ERROR_TYPES[idx]

            print("Please enter a number between 0 and 4.")

        except ValueError:
            print("Invalid input.")


def print_retrieval_metrics(q: dict) -> None:
    """
    Print retrieval metrics if they exist.
    Works with new hybrid results: recall_at_k + average_precision.
    Also tolerates old results: hit + reciprocal_rank.
    """
    print("\n" + "=" * 100)
    print("RETRIEVAL METRICS")
    print("=" * 100)

    if "recall_at_k" in q or "average_precision" in q:
        print(f"Recall@k         : {q.get('recall_at_k', 'N/A')}")
        print(f"Average Precision: {q.get('average_precision', 'N/A')}")
    else:
        print(f"Hit              : {q.get('hit', 'N/A')}")
        print(f"Reciprocal Rank  : {q.get('reciprocal_rank', 'N/A')}")


def print_chunks(q: dict, chunk_store: dict) -> None:
    relevant_ids = set(q.get("relevant_chunk_ids", []))
    retrieved_ids = q.get("retrieved_chunk_ids", [])

    chunk_texts = get_chunk_texts(q, chunk_store)

    if not chunk_texts:
        print("\nNO RETRIEVED CHUNKS")
        return

    print("\n" + "=" * 100)
    print("RETRIEVED CONTEXT")
    print("=" * 100)

    for i, (cid, text) in enumerate(zip(retrieved_ids, chunk_texts)):
        marker = "✓ RELEVANT" if cid in relevant_ids else "✗"

        print("\n" + "-" * 100)
        print(f"CHUNK [{i + 1}] | {cid} | {marker}")
        print("-" * 100)

        if CHUNK_PREVIEW_LEN is None:
            print(text)
        else:
            preview = text[:CHUNK_PREVIEW_LEN]

            if len(text) > CHUNK_PREVIEW_LEN:
                preview += "..."

            print(preview)


def get_judge_scores(q: dict) -> dict:
    """
    Safely get judge scores.
    If judge_scores is None, return empty dict.
    """
    scores = q.get("judge_scores")
    return scores if isinstance(scores, dict) else {}


def main() -> None:
    print("\nLoading data...\n")

    results = load_json(RESULTS_PATH)
    chunk_store = load_chunk_store(CHUNK_STORE_PATH)

    existing = []

    if os.path.exists(OUTPUT_PATH):
        existing = load_json(OUTPUT_PATH)
        print(f"Found {len(existing)} already reviewed questions.\n")

    existing_ids = {e["id"] for e in existing}

    sampled = sample_questions(
        results,
        SAMPLES_PER_TYPE,
        RANDOM_SEED,
    )

    remaining = [
        q for q in sampled
        if q["id"] not in existing_ids
    ]

    print(f"TOTAL QUESTIONS : {len(sampled)}")
    print(f"ALREADY REVIEWED: {len(existing_ids)}")
    print(f"REMAINING       : {len(remaining)}")

    print("\nPress ENTER to continue...")
    input()

    manual_results = list(existing)

    for i, q in enumerate(remaining):
        print("\n" + "=" * 120)
        print(f"[{i + 1}/{len(remaining)}]  ID={q['id']}  TYPE={q['type']}")
        print("=" * 120)

        print("\nQUESTION:")
        print(q["question"])

        print("\nREFERENCE ANSWER:")
        print(q["reference_answer"])

        print("\nKEY FACTS:")
        for kf in q["key_facts"]:
            print(f"  - {kf}")

        print_retrieval_metrics(q)
        print_chunks(q, chunk_store)

        print("\n" + "=" * 100)
        print("GENERATED ANSWER")
        print("=" * 100)
        print(q["generated_answer"])

        judge_scores = get_judge_scores(q)

        print("\n" + "=" * 100)
        print("CLAUDE SCORES")
        print("=" * 100)

        print(
            f"Accuracy={judge_scores.get('accuracy', 'N/A')} | "
            f"Groundedness={judge_scores.get('groundedness', 'N/A')} | "
            f"Relevance={judge_scores.get('relevance', 'N/A')} | "
            f"Completeness={judge_scores.get('completeness', 'N/A')} | "
            f"Error={judge_scores.get('error_type', 'N/A')}"
        )

        reasoning = judge_scores.get("reasoning", "")

        if reasoning:
            print("\nCLAUDE REASONING:")
            print(reasoning)

        print("\n" + "=" * 100)
        print("YOUR SCORES (1-5)")
        print("=" * 100)

        accuracy = get_score("Accuracy     : ")
        groundedness = get_score("Groundedness : ")
        relevance = get_score("Relevance    : ")
        completeness = get_score("Completeness : ")

        error_type = get_error_type()

        notes = input("\nNotes (optional): ").strip()

        manual_results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "generated_answer": q["generated_answer"],
            "reference_answer": q["reference_answer"],
            "key_facts": q["key_facts"],

            # Important fields for later analysis
            "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
            "retrieved_chunk_ids": q.get("retrieved_chunk_ids", []),

            # New retrieval metrics
            "recall_at_k": q.get("recall_at_k"),
            "average_precision": q.get("average_precision"),

            # Optional old metrics, only if present
            "hit": q.get("hit"),
            "reciprocal_rank": q.get("reciprocal_rank"),

            "judge_scores": judge_scores,
            "manual_scores": {
                "accuracy": accuracy,
                "groundedness": groundedness,
                "relevance": relevance,
                "completeness": completeness,
                "error_type": error_type,
                "notes": notes,
            },
        })

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                manual_results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n✓ SAVED ({len(manual_results)}/{len(sampled)})")

    print("\n" + "=" * 120)
    print("MANUAL REVIEW COMPLETE")
    print("=" * 120)

    print(f"\nTotal reviewed: {len(manual_results)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()