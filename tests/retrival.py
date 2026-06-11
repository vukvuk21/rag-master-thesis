import json
from collections import defaultdict
from typing import List, Dict, Any


RESULTS_PATH = "experiments/results/results_final_baseline.json"
OUTPUT_PATH = "experiments/results/metrics_final_baseline.json"


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


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall_recall = []
    overall_ap = []

    by_type = defaultdict(lambda: {
        "recall_at_k": [],
        "average_precision": [],
    })

    detailed_results = []

    for item in results:
        qid = item["id"]
        qtype = item["type"]

        retrieved_ids = item["retrieved_chunk_ids"]
        relevant_ids = item["relevant_chunk_ids"]

        rec = recall_at_k(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)

        overall_recall.append(rec)
        overall_ap.append(ap)

        by_type[qtype]["recall_at_k"].append(rec)
        by_type[qtype]["average_precision"].append(ap)

        detailed_results.append({
            "id": qid,
            "type": qtype,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "recall_at_k": rec,
            "average_precision": ap,
        })

    return {
        "overall": {
            "n_questions": len(results),
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
    }


def main():
    results = load_json(RESULTS_PATH)
    metrics = compute_metrics(results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Retrieval metrics computed")
    print(f"Results input : {RESULTS_PATH}")
    print(f"Output        : {OUTPUT_PATH}")
    print("=" * 60)

    print("\nOverall:")
    print(f"Total questions: {metrics['overall']['n_questions']}")
    print(f"Recall@k       : {metrics['overall']['recall_at_k']:.4f}")
    print(f"MAP            : {metrics['overall']['map']:.4f}")

    print("\nBy type:")
    for qtype, vals in metrics["by_type"].items():
        print(
            f"{qtype:<15} "
            f"n={vals['n']:<3} "
            f"Recall@k={vals['recall_at_k']:.4f} "
            f"MAP={vals['map']:.4f}"
        )


if __name__ == "__main__":
    main()