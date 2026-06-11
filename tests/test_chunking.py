import json
from collections import Counter

with open("data/raw/history_corpus_clean.json", encoding="utf-8") as f:
    articles = json.load(f)

counts = Counter(a["source_category"] for a in articles)
for cat, count in sorted(counts.items()):
    print(f"{cat:<35} {count}")