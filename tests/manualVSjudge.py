import json
from collections import defaultdict


# ============================================================
# OVDE MENJAŠ SAMO PUTANJE DO MANUAL FAJLOVA
# ============================================================

CONFIGS = [
    {
        "name": "baseline k=10 512/50",
        "manual_path": "experiments/results/manual_K=10.json",
    },
    {
        "name": "large k=10 512/50",
        "manual_path": "experiments/results/manual_K=10_large.json",
    },
    {
        "name": "bge-m3 k=10 512/50",
        "manual_path": "experiments/results/manual_K=10_bge.json",
    },
    {
        "name": "hybrid bge-m3 + bm25 + rrf + reranker k=10",
        "manual_path": "experiments/results/manual_hybrid.json",
    },
    {
        "name": "baseline k=5 512/50",
        "manual_path": "experiments/results/manual_baseline.json",
    },
    {
        "name": "baseline k=3 512/50",
        "manual_path": "experiments/results/manual_K=3.json",
    },
    {
        "name": "contextual chunking k=5",
        "manual_path": "experiments/results/manual_contextual.json",
    },
    {
        "name": "small chunks k=5",
        "manual_path": "experiments/results/manual_small_chunks.json",
    },
]


DIMS = ["accuracy", "groundedness", "relevance", "completeness"]

ERROR_TYPES_ORDER = [
    "NONE",
    "OMISSION",
    "RETRIEVAL_FAIL",
    "HALLUCINATION",
    "MISATTRIBUTION",
    "UNKNOWN",
]


def avg(values):
    return sum(values) / len(values) if values else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_manual_vs_judge_from_manual_file(manual_path, name):
    data = load_json(manual_path)

    manual_scores = defaultdict(list)
    judge_scores = defaultdict(list)

    manual_errors = defaultdict(int)
    judge_errors = defaultdict(int)

    by_type_manual = defaultdict(lambda: defaultdict(list))
    by_type_judge = defaultdict(lambda: defaultdict(list))

    by_type_manual_errors = defaultdict(lambda: defaultdict(int))
    by_type_judge_errors = defaultdict(lambda: defaultdict(int))

    missing_manual_scores = []
    missing_judge_scores = []

    for row in data:
        qid = row.get("id", "UNKNOWN_ID")
        question_type = row.get("type", "UNKNOWN_TYPE")

        if "manual_scores" not in row:
            missing_manual_scores.append(qid)
            continue

        if "judge_scores" not in row:
            missing_judge_scores.append(qid)
            continue

        manual = row["manual_scores"]
        judge = row["judge_scores"]

        for d in DIMS:
            manual_scores[d].append(manual[d])
            judge_scores[d].append(judge[d])

            by_type_manual[question_type][d].append(manual[d])
            by_type_judge[question_type][d].append(judge[d])

        manual_error = manual.get("error_type", "UNKNOWN")
        judge_error = judge.get("error_type", "UNKNOWN")

        manual_errors[manual_error] += 1
        judge_errors[judge_error] += 1

        by_type_manual_errors[question_type][manual_error] += 1
        by_type_judge_errors[question_type][judge_error] += 1

    overall = {}

    for d in DIMS:
        manual_avg = avg(manual_scores[d])
        judge_avg = avg(judge_scores[d])

        overall[d] = {
            "manual": manual_avg,
            "judge": judge_avg,
            "diff_judge_minus_manual": judge_avg - manual_avg,
        }

    by_type = {}

    all_question_types = sorted(
        set(by_type_manual.keys()) | set(by_type_judge.keys())
    )

    for question_type in all_question_types:
        by_type[question_type] = {
            "scores": {},
            "manual_errors": dict(by_type_manual_errors[question_type]),
            "judge_errors": dict(by_type_judge_errors[question_type]),
        }

        for d in DIMS:
            manual_avg = avg(by_type_manual[question_type][d])
            judge_avg = avg(by_type_judge[question_type][d])

            by_type[question_type]["scores"][d] = {
                "manual": manual_avg,
                "judge": judge_avg,
                "diff_judge_minus_manual": judge_avg - manual_avg,
            }

    return {
        "name": name,
        "path": manual_path,
        "n_questions": len(data),
        "n_used": len(manual_scores["accuracy"]),
        "missing_manual_scores": missing_manual_scores,
        "missing_judge_scores": missing_judge_scores,
        "overall": overall,
        "manual_errors": dict(manual_errors),
        "judge_errors": dict(judge_errors),
        "by_type": by_type,
    }


def print_one_summary(summary):
    print("\n" + "=" * 110)
    print(summary["name"])
    print("=" * 110)

    print(f"Path       : {summary['path']}")
    print(f"Questions  : {summary['n_questions']}")
    print(f"Used       : {summary['n_used']}")

    if summary["missing_manual_scores"]:
        print(f"\nMissing manual_scores: {len(summary['missing_manual_scores'])}")
        print(summary["missing_manual_scores"])

    if summary["missing_judge_scores"]:
        print(f"\nMissing judge_scores: {len(summary['missing_judge_scores'])}")
        print(summary["missing_judge_scores"])

    print("\nOVERALL — MANUAL VS LLM-AS-JUDGE")
    print("-" * 75)
    print(f"{'Metric':<15} {'Manual':>10} {'Judge':>10} {'Diff(J-M)':>12}")
    print("-" * 75)

    for d in DIMS:
        row = summary["overall"][d]
        print(
            f"{d:<15} "
            f"{row['manual']:>10.2f} "
            f"{row['judge']:>10.2f} "
            f"{row['diff_judge_minus_manual']:>12.2f}"
        )

    print("\nERROR DISTRIBUTION")
    print("-" * 75)
    print(f"Manual errors: {summary['manual_errors']}")
    print(f"Judge errors : {summary['judge_errors']}")

    print("\nBY QUESTION TYPE")
    print("-" * 75)

    for question_type, type_data in summary["by_type"].items():
        print(f"\n{question_type}")
        print(f"{'Metric':<15} {'Manual':>10} {'Judge':>10} {'Diff(J-M)':>12}")
        print("-" * 75)

        for d in DIMS:
            row = type_data["scores"][d]
            print(
                f"{d:<15} "
                f"{row['manual']:>10.2f} "
                f"{row['judge']:>10.2f} "
                f"{row['diff_judge_minus_manual']:>12.2f}"
            )

        print(f"Manual errors by type: {type_data['manual_errors']}")
        print(f"Judge errors by type : {type_data['judge_errors']}")


def print_global_comparison_table(all_summaries):
    print("\n\n" + "#" * 140)
    print("GLOBAL COMPARISON — OVERALL SCORES")
    print("#" * 140)

    header = (
        f"{'Config':<52} "
        f"{'M Acc':>6} {'J Acc':>6} {'ΔAcc':>6} "
        f"{'M Grd':>6} {'J Grd':>6} {'ΔGrd':>6} "
        f"{'M Rel':>6} {'J Rel':>6} {'ΔRel':>6} "
        f"{'M Cmp':>6} {'J Cmp':>6} {'ΔCmp':>6}"
    )

    print(header)
    print("-" * 140)

    for s in all_summaries:
        o = s["overall"]

        print(
            f"{s['name']:<52} "
            f"{o['accuracy']['manual']:>6.2f} "
            f"{o['accuracy']['judge']:>6.2f} "
            f"{o['accuracy']['diff_judge_minus_manual']:>6.2f} "
            f"{o['groundedness']['manual']:>6.2f} "
            f"{o['groundedness']['judge']:>6.2f} "
            f"{o['groundedness']['diff_judge_minus_manual']:>6.2f} "
            f"{o['relevance']['manual']:>6.2f} "
            f"{o['relevance']['judge']:>6.2f} "
            f"{o['relevance']['diff_judge_minus_manual']:>6.2f} "
            f"{o['completeness']['manual']:>6.2f} "
            f"{o['completeness']['judge']:>6.2f} "
            f"{o['completeness']['diff_judge_minus_manual']:>6.2f}"
        )


def print_global_error_table(all_summaries):
    print("\n\n" + "#" * 140)
    print("GLOBAL COMPARISON — ERROR DISTRIBUTION")
    print("#" * 140)

    all_error_types = set()

    for s in all_summaries:
        all_error_types.update(s["manual_errors"].keys())
        all_error_types.update(s["judge_errors"].keys())

    error_types = [
        e for e in ERROR_TYPES_ORDER if e in all_error_types
    ] + sorted(all_error_types - set(ERROR_TYPES_ORDER))

    header = f"{'Config':<52}"

    for e in error_types:
        short = e[:6]
        header += f" {'M_' + short:>8} {'J_' + short:>8}"

    print(header)
    print("-" * 140)

    for s in all_summaries:
        row = f"{s['name']:<52}"

        for e in error_types:
            row += f" {s['manual_errors'].get(e, 0):>8} {s['judge_errors'].get(e, 0):>8}"

        print(row)


def print_ranking_by_manual_average(all_summaries):
    print("\n\n" + "#" * 140)
    print("RANKING BY MANUAL AVERAGE SCORE")
    print("#" * 140)

    ranked = []

    for s in all_summaries:
        o = s["overall"]

        manual_avg_score = avg([
            o["accuracy"]["manual"],
            o["groundedness"]["manual"],
            o["relevance"]["manual"],
            o["completeness"]["manual"],
        ])

        judge_avg_score = avg([
            o["accuracy"]["judge"],
            o["groundedness"]["judge"],
            o["relevance"]["judge"],
            o["completeness"]["judge"],
        ])

        ranked.append({
            "name": s["name"],
            "manual_avg": manual_avg_score,
            "judge_avg": judge_avg_score,
            "manual_errors": s["manual_errors"],
            "judge_errors": s["judge_errors"],
            "overall": o,
        })

    ranked.sort(key=lambda x: x["manual_avg"], reverse=True)

    print(
        f"{'Rank':>4} {'Config':<52} "
        f"{'ManualAvg':>10} {'JudgeAvg':>10} "
        f"{'M_Acc':>6} {'M_Grd':>6} {'M_Rel':>6} {'M_Cmp':>6} "
        f"{'J_Acc':>6} {'J_Grd':>6} {'J_Rel':>6} {'J_Cmp':>6}"
    )

    print("-" * 140)

    for i, row in enumerate(ranked, start=1):
        o = row["overall"]

        print(
            f"{i:>4} {row['name']:<52} "
            f"{row['manual_avg']:>10.2f} "
            f"{row['judge_avg']:>10.2f} "
            f"{o['accuracy']['manual']:>6.2f} "
            f"{o['groundedness']['manual']:>6.2f} "
            f"{o['relevance']['manual']:>6.2f} "
            f"{o['completeness']['manual']:>6.2f} "
            f"{o['accuracy']['judge']:>6.2f} "
            f"{o['groundedness']['judge']:>6.2f} "
            f"{o['relevance']['judge']:>6.2f} "
            f"{o['completeness']['judge']:>6.2f}"
        )


def run_all(configs):
    all_summaries = []

    for cfg in configs:
        try:
            summary = compute_manual_vs_judge_from_manual_file(
                manual_path=cfg["manual_path"],
                name=cfg["name"],
            )

            all_summaries.append(summary)
            print_one_summary(summary)

        except FileNotFoundError as e:
            print("\n" + "=" * 110)
            print(f"SKIPPED: {cfg['name']}")
            print("=" * 110)
            print(f"File not found: {e.filename}")

        except KeyError as e:
            print("\n" + "=" * 110)
            print(f"ERROR IN: {cfg['name']}")
            print("=" * 110)
            print(f"Missing key: {e}")
            print("Proveri da li svaki zapis ima judge_scores i manual_scores sa istim poljima.")

    if all_summaries:
        print_global_comparison_table(all_summaries)
        print_global_error_table(all_summaries)
        print_ranking_by_manual_average(all_summaries)

    return all_summaries


all_summaries = run_all(CONFIGS)