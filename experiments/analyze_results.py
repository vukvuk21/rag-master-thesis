import json
from collections import defaultdict

RESULTS_PATH = "experiments/results/results_baseline.json"

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

by_type = defaultdict(lambda: defaultdict(list))

for r in results:
    t = r["type"]
    by_type[t]["hit"].append(r["hit"])
    by_type[t]["rr"].append(r["reciprocal_rank"])
    if r["judge_scores"]:
        for dim in ["accuracy", "groundedness", "relevance", "completeness"]:
            by_type[t][dim].append(r["judge_scores"].get(dim, 0))

print(f"{'Type':<20} {'N':>4} {'Hit':>6} {'MRR':>6} {'Acc':>6} {'Grd':>6} {'Rel':>6} {'Cmp':>6}")
print("-" * 60)

for t, v in sorted(by_type.items()):
    print(f"{t:<20} {len(v['hit']):>4} "
          f"{avg(v['hit']):>6.3f} "
          f"{avg(v['rr']):>6.3f} "
          f"{avg(v['accuracy']):>6.2f} "
          f"{avg(v['groundedness']):>6.2f} "
          f"{avg(v['relevance']):>6.2f} "
          f"{avg(v['completeness']):>6.2f}")