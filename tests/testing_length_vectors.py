import json
from pathlib import Path

paths = [
    "data/embeddings/embeddings_baseline.json",
    "data/embeddings/embeddings_baseline.large.json",
    "data/embeddings/embeddings_baseline_bge_m3.json",
]

for path in paths:
    p = Path(path)

    if not p.exists():
        print(f"{path}: fajl ne postoji")
        continue

    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload

    dims = {len(item["embedding"]) for item in data[:20]}

    print(path)
    print("  broj segmenata:", len(data))
    print("  dimenzije u prvih 20 embedding-a:", dims)