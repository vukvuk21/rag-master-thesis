import json

CHUNK_STORE_PATH = "data/chunks/chunks_contextual.json"


def load_chunks(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c["text"] for c in chunks}


def search_claim_evidence_in_chunks(chunk_ids, claims, chunk_store, context_window=220):
    for claim in claims:
        print("\n" + "=" * 100)
        print(f"CLAIM: {claim['claim']}")
        print("=" * 100)

        found_any = False

        for evidence in claim["evidence_terms"]:
            evidence_lower = evidence.lower()
            evidence_found = False

            print(f"\nEvidence term: {evidence}")

            for cid in chunk_ids:
                text = chunk_store.get(cid)
                if text is None:
                    continue

                text_lower = text.lower()
                pos = text_lower.find(evidence_lower)

                if pos != -1:
                    found_any = True
                    evidence_found = True

                    start = max(0, pos - context_window)
                    end = min(len(text), pos + len(evidence) + context_window)

                    print(f"FOUND in chunk: {cid}")
                    print(f"...{text[start:end]}...")

            if not evidence_found:
                print("NOT FOUND")

        if not found_any:
            print("\nCLAIM STATUS: likely NOT GROUNDED")
        else:
            print("\nCLAIM STATUS: evidence found, needs semantic check")


if __name__ == "__main__":
    chunk_store = load_chunks(CHUNK_STORE_PATH)

    chunk_ids = [
    "68793880_0001",
    "68793880_0007",
    "68793880_0011",
]

claims = [
    {
        "claim": "Marmaduke Johnson was the first printer in the American colonies to operate his own press.",
        "evidence_terms": [
            "Marmaduke Johnson",
            "first printer",
            "operate his own press",
            "American colonies",
        ],
    },
    {
        "claim": "Johnson began his enterprise in 1674.",
        "evidence_terms": [
            "1674",
            "enterprise",
            "began",
        ],
    },
    {
        "claim": "Johnson died shortly after beginning his enterprise.",
        "evidence_terms": [
            "died shortly after",
            "died",
            "shortly after",
        ],
    },

    # ⚠️ potentially ungrounded embellishment
    {
        "claim": "Johnson never fulfilled his dream of running his own private enterprise.",
        "evidence_terms": [
            "fulfilled his dream",
            "private enterprise",
        ],
    },
]

search_claim_evidence_in_chunks(chunk_ids, claims, chunk_store)