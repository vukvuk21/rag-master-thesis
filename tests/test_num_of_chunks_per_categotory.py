import json

BASELINE_CHUNKS_PATH = "data/chunks/chunks_baseline.json"
CONTEXTUAL_CHUNKS_PATH = "data/chunks/chunks_contextual.json"

BASELINE_CHUNK_ID = "80901075_0026"
MAPPED_CONTEXTUAL_IDS = [
    "80901075_0042",
    "80901075_0041"
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_by_id(chunks):
    return {c["chunk_id"]: c for c in chunks}

baseline_chunks = load_json(BASELINE_CHUNKS_PATH)
contextual_chunks = load_json(CONTEXTUAL_CHUNKS_PATH)

baseline_by_id = index_by_id(baseline_chunks)
contextual_by_id = index_by_id(contextual_chunks)

baseline_chunk = baseline_by_id.get(BASELINE_CHUNK_ID)

print("=" * 100)
print("BASELINE CHUNK")
print("=" * 100)

if baseline_chunk is None:
    print(f"Baseline chunk not found: {BASELINE_CHUNK_ID}")
else:
    print(f"chunk_id: {baseline_chunk['chunk_id']}")
    print(f"doc_id: {baseline_chunk['doc_id']}")
    print(f"title: {baseline_chunk.get('title')}")
    print(f"chunk_index: {baseline_chunk.get('chunk_index')}")
    print(f"token_count: {baseline_chunk.get('token_count')}")
    print("-" * 100)
    print(baseline_chunk["text"])

print("\n\n" + "=" * 100)
print("MAPPED CONTEXTUAL CHUNKS")
print("=" * 100)

for cid in MAPPED_CONTEXTUAL_IDS:
    chunk = contextual_by_id.get(cid)

    print("\n" + "#" * 100)
    print(f"CONTEXTUAL CHUNK: {cid}")
    print("#" * 100)

    if chunk is None:
        print("NOT FOUND")
        continue

    print(f"chunk_id: {chunk['chunk_id']}")
    print(f"doc_id: {chunk['doc_id']}")
    print(f"title: {chunk.get('title')}")
    print(f"chunk_index: {chunk.get('chunk_index')}")
    print(f"token_count: {chunk.get('token_count')}")
    print("-" * 100)
    print(chunk["text"])