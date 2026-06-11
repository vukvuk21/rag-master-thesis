import json

CHUNK_STORE_PATH = "data/chunks/chunks_baseline.json"

def load_chunks(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c["text"] for c in chunks}

def search_in_chunks(chunk_ids: list, keyword: str, chunk_store: dict) -> None:
    """Pretraži ključnu reč u zadatim chunkovima."""
    keyword_lower = keyword.lower()
    found_any = False

    for cid in chunk_ids:
        text = chunk_store.get(cid, None)
        if text is None:
            print(f"[{cid}] NOT FOUND in chunk store")
            continue

        if keyword_lower in text.lower():
            found_any = True
            # Pronađi sve pozicije i prikaži kontekst oko njih
            idx = 0
            occurrences = []
            while True:
                pos = text.lower().find(keyword_lower, idx)
                if pos == -1:
                    break
                start = max(0, pos - 150)
                end = min(len(text), pos + 150)
                occurrences.append(text[start:end])
                idx = pos + 1

            print(f"\n[{cid}] — '{keyword}' found {len(occurrences)} time(s):")
            for i, snippet in enumerate(occurrences):
                print(f"  ...{snippet}...")
                print()

    if not found_any:
        print(f"\nKeyword '{keyword}' not found in any of the provided chunks.")


if __name__ == "__main__":
    chunk_store = load_chunks(CHUNK_STORE_PATH)

chunk_ids = [
    "34634219_0011",
    "34634219_0013",
    "34634219_0014",
    "34634219_0012",
    "34634219_0009",
]
keyword = "second category"
print(f"Searching for '{keyword}' in {len(chunk_ids)} chunks...\n")
search_in_chunks(chunk_ids, keyword, chunk_store)