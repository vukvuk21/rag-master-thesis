import json
from collections import defaultdict

PATH = "data/evaluation/eval_dataset_enriched.json"

with open(PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

by_type = defaultdict(int)
by_category = defaultdict(int)

for q in questions:
    by_type[q["type"]] += 1
    cat = q["source_categories"][0] if q.get("source_categories") else "unknown"
    by_category[cat] += 1

print(f"Total: {len(questions)}")
print(f"\nBy type:")
for t, c in sorted(by_type.items()):
    print(f"  {t:<20} {c}")

print(f"\nBy category:")
for cat, c in sorted(by_category.items()):
    print(f"  {cat:<35} {c}")