import json
import re
from collections import defaultdict
from typing import Any, Dict, List


BASELINE_RESULTS_PATH = "experiments/results/results1_baseline.json"
BASELINE_CHUNKS_PATH = "data/chunks/chunks_baseline.json"
TARGET_CHUNKS_PATH = "data/chunks/chunks_contextual.json"

OUTPUT_PATH = "experiments/results/unmapped_contextual_candidates.json"

UNMAPPED_QUESTION_IDS = [
    "q_single_003",
    "q_single_014",
    "q_single_033",
    "q_single_053",
    "q_multi_003",
    "q_complex_012",
    "q_complex_018",
    "q_complex_022",
    "q_complex_053",
]

TOP_N = 10


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def index_by_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["chunk_id"]: item for item in items}


def index_results_by_id(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in results}


def group_chunks_by_doc_id(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_id"]].append(chunk)
    return grouped


def main() -> None:
    baseline_results = load_json(BASELINE_RESULTS_PATH)
    baseline_chunks = load_json(BASELINE_CHUNKS_PATH)
    target_chunks = load_json(TARGET_CHUNKS_PATH)

    results_by_id = index_results_by_id(baseline_results)
    baseline_chunks_by_id = index_by_id(baseline_chunks)
    target_chunks_by_doc_id = group_chunks_by_doc_id(target_chunks)

    output = {}

    for qid in UNMAPPED_QUESTION_IDS:
        question = results_by_id[qid]

        question_output = {
            "id": qid,
            "type": question["type"],
            "question": question["question"],
            "baseline_relevant_chunk_ids": question["relevant_chunk_ids"],
            "baseline_chunks": [],
        }

        print("\n" + "=" * 100)
        print(f"{qid} | {question['type']}")
        print(question["question"])
        print("=" * 100)

        for baseline_id in question["relevant_chunk_ids"]:
            baseline_chunk = baseline_chunks_by_id[baseline_id]
            doc_id = baseline_chunk["doc_id"]
            candidates = target_chunks_by_doc_id.get(doc_id, [])

            scored = []
            for target_chunk in candidates:
                score = token_overlap_ratio(
                    baseline_chunk["text"],
                    target_chunk["text"],
                )

                scored.append({
                    "chunk_id": target_chunk["chunk_id"],
                    "chunk_index": target_chunk.get("chunk_index"),
                    "title": target_chunk.get("title"),
                    "overlap_ratio": score,
                    "token_count": target_chunk.get("token_count"),
                    "text": target_chunk["text"],
                })

            scored.sort(key=lambda x: x["overlap_ratio"], reverse=True)
            top_candidates = scored[:TOP_N]

            question_output["baseline_chunks"].append({
                "baseline_chunk_id": baseline_id,
                "doc_id": doc_id,
                "title": baseline_chunk.get("title"),
                "baseline_chunk_index": baseline_chunk.get("chunk_index"),
                "baseline_text": baseline_chunk["text"],
                "top_candidates": top_candidates,
            })

            print(f"\nBASELINE CHUNK: {baseline_id}")
            print(f"doc_id: {doc_id}")
            print(f"title : {baseline_chunk.get('title')}")
            print(f"index : {baseline_chunk.get('chunk_index')}")
            print("-" * 100)
            print(baseline_chunk["text"])

            print("\nTOP CONTEXTUAL CANDIDATES")
            print("-" * 100)

            for i, cand in enumerate(top_candidates, start=1):
                print(f"\n#{i}")
                print(f"chunk_id      : {cand['chunk_id']}")
                print(f"chunk_index   : {cand['chunk_index']}")
                print(f"overlap_ratio : {cand['overlap_ratio']:.4f}")
                print(f"token_count   : {cand['token_count']}")
                print("-" * 100)
                print(cand["text"])

        output[qid] = question_output

    save_json(output, OUTPUT_PATH)

    print("\n" + "=" * 100)
    print(f"Saved candidates to: {OUTPUT_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()