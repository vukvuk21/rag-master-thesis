import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any


# ============================================================
# OVDE MENJAŠ PUTANJE I IMENA KONFIGURACIJA
# ============================================================

CONFIGS = [
    {
        "experiment": "k=10 512/50",
        "results_path": "experiments/results/resultsK=10_baseline.json",
        "output_path": "experiments/results/metrics_final_K=10_baseline_full.json",
        "config": {
            "embedder": "text-embedding-3-small",
            "k": 10,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
        },
    },
    {
        "experiment": "large k=10 512/50",
        "results_path": "experiments/results/resultsK=10_large.json",
        "output_path": "experiments/results/metrics_final_K=10_large_full.json",
        "config": {
            "embedder": "text-embedding-3-large",
            "k": 10,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
        },
    },
    {
        "experiment": "bge-m3 k=10 512/50",
        "results_path": "experiments/results/resultsK=10_bge_m3.json",
        "output_path": "experiments/results/metrics_final_K=10_bge_m3_full.json",
        "config": {
            "embedder": "BAAI/bge-m3",
            "embedding_type": "dense_vecs",
            "k": 10,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
            "chroma_path": "data/chroma/bge_m3",
            "collection_name": "history_bge_m3",
        },
    },
    {
        "experiment": "hybrid bge-m3 + bm25 + rrf + reranker k=10",
        "results_path": "experiments/results/results_hybrid_bge_m3_bm25_rerank_k10.json",
        "output_path": "experiments/results/metrics_final_hybrid_bge_m3_bm25_rerank_k10_full.json",
        "config": {
            "retrieval_type": "hybrid_dense_bm25_rrf_rerank",
            "dense_embedder": "BAAI/bge-m3",
            "sparse_retriever": "BM25",
            "fusion": "Reciprocal Rank Fusion",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "final_k": 10,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
        },
    },
    {
        "experiment": "baseline k=5 512/50",
        "results_path": "experiments/results/results_final_baseline.json",
        "output_path": "experiments/results/metrics_final_baseline_full.json",
        "config": {
            "embedder": "text-embedding-3-small",
            "k": 5,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
        },
    },
    {
        "experiment": "baseline k=3 512/50",
        "results_path": "experiments/results/resultsK=3_baseline.json",
        "output_path": "experiments/results/metrics_final_K=3_baseline_full.json",
        "config": {
            "embedder": "text-embedding-3-small",
            "k": 3,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
        },
    },
    {
        "experiment": "contextual chunking k=5",
        "results_path": "experiments/results/results1_contextual.json",
        "output_path": "experiments/results/metrics_final_contextual_full.json",
        "config": {
            "embedder": "text-embedding-3-small",
            "k": 5,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 512,
            "overlap": 50,
            "chunking": "contextual",
        },
    },
    {
        "experiment": "small chunks k=5",
        "results_path": "experiments/results/results1_small_chunks.json",
        "output_path": "experiments/results/metrics_final_small_chunks_full.json",
        "config": {
            "embedder": "text-embedding-3-small",
            "k": 5,
            "generator": "gpt-4o-mini",
            "judge": "claude-opus-4-5",
            "chunk_size": 256,
            "overlap": 25,
        },
    },
]


GLOBAL_OUTPUT_PATH = "experiments/results/all_final_metrics_by_type.json"

DIMS = ["accuracy", "groundedness", "relevance", "completeness"]

QUESTION_TYPE_ORDER = ["single_chunk", "multi_chunk", "complex"]

ERROR_TYPE_ORDER = [
    "NONE",
    "OMISSION",
    "RETRIEVAL_FAIL",
    "HALLUCINATION",
    "MISATTRIBUTION",
    "UNKNOWN",
]


# ============================================================
# METRIC FUNCTIONS
# ============================================================

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


def save_json(data: Any, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_error_distribution(error_counts: Dict[str, int]) -> Dict[str, int]:
    result = {}

    for error_type in ERROR_TYPE_ORDER:
        if error_type in error_counts:
            result[error_type] = error_counts[error_type]

    for error_type in sorted(error_counts.keys()):
        if error_type not in result:
            result[error_type] = error_counts[error_type]

    return result


# ============================================================
# CORE COMPUTATION
# ============================================================

def compute_metrics_for_results(
    results: List[Dict[str, Any]],
    experiment: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    overall_recall = []
    overall_ap = []
    overall_latency = []
    overall_scores = defaultdict(list)

    skipped = 0

    by_type = defaultdict(lambda: {
        "recall_at_k": [],
        "average_precision": [],
        "latency_seconds": [],
        "accuracy": [],
        "groundedness": [],
        "relevance": [],
        "completeness": [],
        "error_distribution": defaultdict(int),
    })

    for item in results:
        qtype = item.get("type", "unknown")

        retrieved_ids = item.get("retrieved_chunk_ids", [])
        relevant_ids = item.get("relevant_chunk_ids", [])

        rec = recall_at_k(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)

        overall_recall.append(rec)
        overall_ap.append(ap)

        by_type[qtype]["recall_at_k"].append(rec)
        by_type[qtype]["average_precision"].append(ap)

        latency = item.get("latency_seconds")
        if latency is not None:
            overall_latency.append(latency)
            by_type[qtype]["latency_seconds"].append(latency)

        judge = item.get("judge_scores")

        if judge is None:
            skipped += 1
            continue

        for dim in DIMS:
            score = judge.get(dim, 0)
            overall_scores[dim].append(score)
            by_type[qtype][dim].append(score)

        error_type = judge.get("error_type", "UNKNOWN")
        by_type[qtype]["error_distribution"][error_type] += 1

    metrics = {
        "experiment": experiment,
        "config": config,
        "overall": {
            "n_questions": len(results),
            "n_skipped": skipped,

            "recall_at_k": avg(overall_recall),
            "map": avg(overall_ap),

            "accuracy": avg(overall_scores["accuracy"]),
            "groundedness": avg(overall_scores["groundedness"]),
            "relevance": avg(overall_scores["relevance"]),
            "completeness": avg(overall_scores["completeness"]),

            "avg_latency_seconds": round(avg(overall_latency), 2),
        },
        "by_type": {},
    }

    # prvo standardni redosled
    ordered_types = [
        t for t in QUESTION_TYPE_ORDER if t in by_type
    ] + sorted(t for t in by_type.keys() if t not in QUESTION_TYPE_ORDER)

    for qtype in ordered_types:
        values = by_type[qtype]

        metrics["by_type"][qtype] = {
            "n": len(values["recall_at_k"]),

            "recall_at_k": avg(values["recall_at_k"]),
            "map": avg(values["average_precision"]),

            "accuracy": avg(values["accuracy"]),
            "groundedness": avg(values["groundedness"]),
            "relevance": avg(values["relevance"]),
            "completeness": avg(values["completeness"]),

            "avg_latency_seconds": round(avg(values["latency_seconds"]), 2),

            "error_distribution": normalize_error_distribution(
                dict(values["error_distribution"])
            ),
        }

    return metrics


# ============================================================
# PRINTING
# ============================================================

def print_one_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 120)
    print(metrics["experiment"])
    print("=" * 120)

    overall = metrics["overall"]

    print("\nOVERALL")
    print("-" * 120)
    print(f"Questions          : {overall['n_questions']}")
    print(f"Skipped            : {overall['n_skipped']}")
    print(f"Recall@k           : {overall['recall_at_k']:.4f}")
    print(f"MAP                : {overall['map']:.4f}")
    print(f"Accuracy           : {overall['accuracy']:.4f}")
    print(f"Groundedness       : {overall['groundedness']:.4f}")
    print(f"Relevance          : {overall['relevance']:.4f}")
    print(f"Completeness       : {overall['completeness']:.4f}")
    print(f"Avg latency        : {overall['avg_latency_seconds']}s")

    print("\nBY TYPE")
    print("-" * 120)

    for qtype, row in metrics["by_type"].items():
        print(f"\n{qtype}")
        print(f"  n                : {row['n']}")
        print(f"  Recall@k         : {row['recall_at_k']:.4f}")
        print(f"  MAP              : {row['map']:.4f}")
        print(f"  Accuracy         : {row['accuracy']:.4f}")
        print(f"  Groundedness     : {row['groundedness']:.4f}")
        print(f"  Relevance        : {row['relevance']:.4f}")
        print(f"  Completeness     : {row['completeness']:.4f}")
        print(f"  Avg latency      : {row['avg_latency_seconds']}s")
        print(f"  Errors           : {row['error_distribution']}")


def print_global_by_type_table(all_metrics: List[Dict[str, Any]]) -> None:
    print("\n\n" + "#" * 170)
    print("GLOBAL TABLE — BY TYPE")
    print("#" * 170)

    for qtype in QUESTION_TYPE_ORDER:
        print("\n" + "=" * 170)
        print(qtype)
        print("=" * 170)

        header = (
            f"{'Config':<58} "
            f"{'n':>4} "
            f"{'Recall':>8} "
            f"{'MAP':>8} "
            f"{'Acc':>8} "
            f"{'Grd':>8} "
            f"{'Rel':>8} "
            f"{'Cmp':>8} "
            f"{'Latency':>9} "
            f"{'NONE':>6} "
            f"{'OMISS':>6} "
            f"{'RET_FAIL':>8} "
            f"{'HALL':>6} "
            f"{'MISATTR':>8}"
        )

        print(header)
        print("-" * 170)

        for metrics in all_metrics:
            if qtype not in metrics["by_type"]:
                continue

            row = metrics["by_type"][qtype]
            errors = row["error_distribution"]

            print(
                f"{metrics['experiment']:<58} "
                f"{row['n']:>4} "
                f"{row['recall_at_k']:>8.4f} "
                f"{row['map']:>8.4f} "
                f"{row['accuracy']:>8.4f} "
                f"{row['groundedness']:>8.4f} "
                f"{row['relevance']:>8.4f} "
                f"{row['completeness']:>8.4f} "
                f"{str(row['avg_latency_seconds']) + 's':>9} "
                f"{errors.get('NONE', 0):>6} "
                f"{errors.get('OMISSION', 0):>6} "
                f"{errors.get('RETRIEVAL_FAIL', 0):>8} "
                f"{errors.get('HALLUCINATION', 0):>6} "
                f"{errors.get('MISATTRIBUTION', 0):>8}"
            )


def print_global_overall_table(all_metrics: List[Dict[str, Any]]) -> None:
    print("\n\n" + "#" * 150)
    print("GLOBAL TABLE — OVERALL, BEZ ERROR DISTRIBUTION")
    print("#" * 150)

    header = (
        f"{'Config':<58} "
        f"{'n':>4} "
        f"{'Recall':>8} "
        f"{'MAP':>8} "
        f"{'Acc':>8} "
        f"{'Grd':>8} "
        f"{'Rel':>8} "
        f"{'Cmp':>8} "
        f"{'Latency':>9}"
    )

    print(header)
    print("-" * 150)

    for metrics in all_metrics:
        o = metrics["overall"]

        print(
            f"{metrics['experiment']:<58} "
            f"{o['n_questions']:>4} "
            f"{o['recall_at_k']:>8.4f} "
            f"{o['map']:>8.4f} "
            f"{o['accuracy']:>8.4f} "
            f"{o['groundedness']:>8.4f} "
            f"{o['relevance']:>8.4f} "
            f"{o['completeness']:>8.4f} "
            f"{str(o['avg_latency_seconds']) + 's':>9}"
        )


# ============================================================
# MAIN
# ============================================================

def run_all(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_metrics = []

    for cfg in configs:
        experiment = cfg["experiment"]
        results_path = cfg["results_path"]
        output_path = cfg["output_path"]
        config = cfg["config"]

        try:
            results = load_json(results_path)

            metrics = compute_metrics_for_results(
                results=results,
                experiment=experiment,
                config=config,
            )

            save_json(metrics, output_path)

            all_metrics.append(metrics)

            print_one_metrics(metrics)

            print(f"\nSaved: {output_path}")

        except FileNotFoundError:
            print("\n" + "=" * 120)
            print(f"SKIPPED: {experiment}")
            print("=" * 120)
            print(f"File not found: {results_path}")
            print("Proveri putanju u CONFIGS.")

        except KeyError as e:
            print("\n" + "=" * 120)
            print(f"ERROR IN: {experiment}")
            print("=" * 120)
            print(f"Missing key: {e}")

    if all_metrics:
        save_json(all_metrics, GLOBAL_OUTPUT_PATH)
        print_global_overall_table(all_metrics)
        print_global_by_type_table(all_metrics)

        print("\nSaved global metrics:")
        print(GLOBAL_OUTPUT_PATH)

    return all_metrics


if __name__ == "__main__":
    all_metrics = run_all(CONFIGS)