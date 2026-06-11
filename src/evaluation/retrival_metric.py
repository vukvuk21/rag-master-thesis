import json
from collections import defaultdict
from typing import List, Dict, Any

RESULTS_PATH = "experiments/results/results1_contextual.json"
MAPPING_PATH = "experiments/results/relevant_mapping_contextual.json"
OUTPUT_PATH = "experiments/results/metrics_retrieval_contextual_mapped.json"


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    found = sum(1 for rid in relevant_ids if rid in retrieved_set)

    return found / len(relevant_ids)


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    num_hits = 0
    precision_sum = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            num_hits += 1
            precision_at_rank = num_hits / rank
            precision_sum += precision_at_rank

    return precision_sum / len(relevant_set)


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(
    results: List[Dict[str, Any]],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:

    overall_recall = []
    overall_ap = []

    by_type = defaultdict(lambda: {
        "recall_at_k": [],
        "average_precision": [],
    })

    detailed_results = []
    skipped = []

    for item in results:
        qid = item["id"]
        qtype = item["type"]
        retrieved_ids = item["retrieved_chunk_ids"]

        if qid not in mapping:
            skipped.append({
                "id": qid,
                "reason": "question_id not found in mapping"
            })
            continue

        relevant_ids = mapping[qid]["mapped_relevant_chunk_ids"]

        if not relevant_ids:
            skipped.append({
                "id": qid,
                "reason": "no mapped relevant chunks"
            })
            continue

        rec = recall_at_k(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)

        overall_recall.append(rec)
        overall_ap.append(ap)

        by_type[qtype]["recall_at_k"].append(rec)
        by_type[qtype]["average_precision"].append(ap)

        detailed_results.append({
            "id": qid,
            "type": qtype,
            "baseline_relevant_chunk_ids": mapping[qid]["baseline_relevant_chunk_ids"],
            "mapped_relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "recall_at_k": rec,
            "average_precision": ap,
        })

    metrics = {
        "overall": {
            "n_questions_total": len(results),
            "n_questions_evaluated": len(detailed_results),
            "n_skipped": len(skipped),
            "recall_at_k": avg(overall_recall),
            "map": avg(overall_ap),
        },
        "by_type": {
            qtype: {
                "n": len(values["recall_at_k"]),
                "recall_at_k": avg(values["recall_at_k"]),
                "map": avg(values["average_precision"]),
            }
            for qtype, values in by_type.items()
        },
        "details": detailed_results,
        "skipped": skipped,
    }

    return metrics


def main():
    results = load_json(RESULTS_PATH)
    mapping = load_json(MAPPING_PATH)

    metrics = compute_metrics(results, mapping)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Mapped retrieval metrics computed")
    print(f"Results input : {RESULTS_PATH}")
    print(f"Mapping input : {MAPPING_PATH}")
    print(f"Output        : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nOverall:")
    print(f"Total questions    : {metrics['overall']['n_questions_total']}")
    print(f"Evaluated questions: {metrics['overall']['n_questions_evaluated']}")
    print(f"Skipped questions  : {metrics['overall']['n_skipped']}")
    print(f"Recall@k           : {metrics['overall']['recall_at_k']:.4f}")
    print(f"MAP                : {metrics['overall']['map']:.4f}")

    print("\nBy type:")
    for qtype, vals in metrics["by_type"].items():
        print(
            f"{qtype:<15} "
            f"n={vals['n']:<3} "
            f"Recall@k={vals['recall_at_k']:.4f} "
            f"MAP={vals['map']:.4f}"
        )

    if metrics["skipped"]:
        print("\nSkipped:")
        for s in metrics["skipped"]:
            print(f"{s['id']} - {s['reason']}")


if __name__ == "__main__":
    main()