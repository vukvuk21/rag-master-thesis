import json
from collections import defaultdict

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def analyze_manual(path, name):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dims = ["accuracy", "groundedness", "relevance", "completeness"]
    scores = defaultdict(list)
    error_counts = defaultdict(int)
    by_type = defaultdict(lambda: defaultdict(list))

    for r in data:
        m = r["manual_scores"]
        t = r["type"]
        for d in dims:
            scores[d].append(m[d])
            by_type[t][d].append(m[d])
        error_counts[m["error_type"]] += 1

    print(f"\n{'='*50}")
    print(f"{name} — Manual ({len(data)} questions)")
    print(f"{'='*50}")
    print(f"  Accuracy     : {avg(scores['accuracy']):.2f}")
    print(f"  Groundedness : {avg(scores['groundedness']):.2f}")
    print(f"  Relevance    : {avg(scores['relevance']):.2f}")
    print(f"  Completeness : {avg(scores['completeness']):.2f}")
    print(f"  Errors: {dict(error_counts)}")
    print(f"\n  By type:")
    for t, v in sorted(by_type.items()):
        print(f"    {t:<15} Acc:{avg(v['accuracy']):.2f} Grd:{avg(v['groundedness']):.2f} Rel:{avg(v['relevance']):.2f} Cmp:{avg(v['completeness']):.2f}")

analyze_manual("experiments/results/manual_baseline.json", "baseline k=10 512/50")
analyze_manual("experiments/results/manual_K=10_large.json", "large k=10 512/50")
analyze_manual("experiments/results/manual_K=10_bge.json", "bge k=10 512/50")