"""
Enriches the raw evaluation dataset with full chunk texts.
Makes manual review easier — no need to look up chunks by ID.
"""

import json

CHUNKS_PATH = "data/chunks/chunks_baseline.json"
EVAL_INPUT_PATH = "data/evaluation/eval_dataset_raw.json"
EVAL_OUTPUT_PATH = "data/evaluation/eval_dataset_enriched.json"


def main() -> None:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_map = {c["chunk_id"]: c["text"] for c in chunks}

    with open(EVAL_INPUT_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        q["relevant_chunks_text"] = [
            chunk_map.get(cid, "CHUNK NOT FOUND")
            for cid in q["relevant_chunk_ids"]
        ]

    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Enriched {len(questions)} questions -> {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()