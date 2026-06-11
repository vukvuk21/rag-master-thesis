import json
import re
from collections import defaultdict
from typing import Any, Dict, List


# =========================
# CONFIG
# =========================

BASELINE_RESULTS_PATH = "experiments/results/results1_baseline.json"

BASELINE_CHUNKS_PATH = "data/chunks/chunks_baseline.json"
TARGET_CHUNKS_PATH = "data/chunks/chunks_size_256.json"

OUTPUT_MAPPING_PATH = "experiments/results/relevant_mapping_256.json"
OUTPUT_UPDATED_RESULTS_PATH = "experiments/results/results8_contextual_with_mapped_relevant.json"

MIN_OVERLAP_RATIO = 0.52


# =========================
# JSON HELPERS
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# TEXT OVERLAP
# =========================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def token_overlap_ratio(source_text: str, target_text: str) -> float:
    source_tokens = set(tokenize(source_text))
    target_tokens = set(tokenize(target_text))

    if not source_tokens:
        return 0.0

    return len(source_tokens & target_tokens) / len(source_tokens)


# =========================
# INDEXING
# =========================

def index_chunks_by_id(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def group_chunks_by_doc_id(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_id"]].append(chunk)
    return grouped


# =========================
# MAPPING
# =========================

def find_matching_target_chunks(
    baseline_chunk: Dict[str, Any],
    target_chunks_same_doc: List[Dict[str, Any]],
    min_overlap_ratio: float,
) -> List[Dict[str, Any]]:
    matches = []

    for target_chunk in target_chunks_same_doc:
        ratio = token_overlap_ratio(
            baseline_chunk["text"],
            target_chunk["text"]
        )

        if ratio >= min_overlap_ratio:
            matches.append({
                "chunk_id": target_chunk["chunk_id"],
                "chunk_index": target_chunk.get("chunk_index"),
                "overlap_ratio": ratio,
            })

    matches.sort(key=lambda x: x["overlap_ratio"], reverse=True)
    return matches


def build_relevant_mapping(
    baseline_results: List[Dict[str, Any]],
    baseline_chunks_by_id: Dict[str, Dict[str, Any]],
    target_chunks_by_doc_id: Dict[str, List[Dict[str, Any]]],
    min_overlap_ratio: float,
) -> Dict[str, Any]:
    mapping = {}

    for item in baseline_results:
        qid = item["id"]

        mapped_ids = []
        mapping_details = {}

        for baseline_relevant_id in item["relevant_chunk_ids"]:
            baseline_chunk = baseline_chunks_by_id.get(baseline_relevant_id)

            if baseline_chunk is None:
                mapping_details[baseline_relevant_id] = {
                    "error": "baseline relevant chunk not found",
                    "mapped_target_chunk_ids": []
                }
                continue

            doc_id = baseline_chunk["doc_id"]
            target_candidates = target_chunks_by_doc_id.get(doc_id, [])

            matches = find_matching_target_chunks(
                baseline_chunk=baseline_chunk,
                target_chunks_same_doc=target_candidates,
                min_overlap_ratio=min_overlap_ratio,
            )

            matched_ids = [m["chunk_id"] for m in matches]
            mapped_ids.extend(matched_ids)

            mapping_details[baseline_relevant_id] = {
                "doc_id": doc_id,
                "title": baseline_chunk.get("title"),
                "baseline_chunk_index": baseline_chunk.get("chunk_index"),
                "mapped_target_chunk_ids": matched_ids,
                "matches": matches,
            }

        seen = set()
        unique_mapped_ids = []
        for cid in mapped_ids:
            if cid not in seen:
                unique_mapped_ids.append(cid)
                seen.add(cid)

        mapping[qid] = {
            "id": qid,
            "type": item["type"],
            "question": item.get("question"),
            "baseline_relevant_chunk_ids": item["relevant_chunk_ids"],
            "mapped_relevant_chunk_ids": unique_mapped_ids,
            "mapping_details": mapping_details,
        }

    return mapping


# =========================
# OPTIONAL: UPDATE TARGET RESULTS
# =========================

def update_target_results_with_mapping(
    target_results: List[Dict[str, Any]],
    mapping: Dict[str, Any],
) -> List[Dict[str, Any]]:
    updated = []

    for item in target_results:
        qid = item["id"]
        new_item = dict(item)

        if qid in mapping:
            new_item["baseline_relevant_chunk_ids"] = mapping[qid]["baseline_relevant_chunk_ids"]
            new_item["original_relevant_chunk_ids"] = item.get("relevant_chunk_ids", [])
            new_item["relevant_chunk_ids"] = mapping[qid]["mapped_relevant_chunk_ids"]

        updated.append(new_item)

    return updated


# =========================
# MAIN
# =========================

def main() -> None:
    baseline_results = load_json(BASELINE_RESULTS_PATH)
    baseline_chunks = load_json(BASELINE_CHUNKS_PATH)
    target_chunks = load_json(TARGET_CHUNKS_PATH)

    baseline_chunks_by_id = index_chunks_by_id(baseline_chunks)
    target_chunks_by_doc_id = group_chunks_by_doc_id(target_chunks)

    mapping = build_relevant_mapping(
        baseline_results=baseline_results,
        baseline_chunks_by_id=baseline_chunks_by_id,
        target_chunks_by_doc_id=target_chunks_by_doc_id,
        min_overlap_ratio=MIN_OVERLAP_RATIO,
    )

    save_json(mapping, OUTPUT_MAPPING_PATH)

    total_questions = len(mapping)
    unmapped_questions = [
        q for q in mapping.values()
        if len(q["mapped_relevant_chunk_ids"]) == 0
    ]

    print("=" * 60)
    print("Mapping finished")
    print(f"Saved mapping to: {OUTPUT_MAPPING_PATH}")
    print(f"Total questions: {total_questions}")
    print(f"Questions with no mapped chunks: {len(unmapped_questions)}")

    if unmapped_questions:
        print("\nUNMAPPED QUESTIONS")
        print("-" * 60)

        for q in unmapped_questions:
            print(f"\nID      : {q['id']}")
            print(f"TYPE    : {q['type']}")
            print(f"QUESTION: {q['question']}")
            print(f"BASELINE RELEVANT IDS: {q['baseline_relevant_chunk_ids']}")

    print("=" * 60)

    # Ako želiš odmah da napraviš updated results fajl,
    # odkomentariši ovaj deo i podesi TARGET_RESULTS_PATH.
    """
    TARGET_RESULTS_PATH = "experiments/results/results8_contextual.json"

    target_results = load_json(TARGET_RESULTS_PATH)
    updated_results = update_target_results_with_mapping(target_results, mapping)

    save_json(updated_results, OUTPUT_UPDATED_RESULTS_PATH)
    print(f"Saved updated results to: {OUTPUT_UPDATED_RESULTS_PATH}")
    """


if __name__ == "__main__":
    main()