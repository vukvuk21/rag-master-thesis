"""
Baseline chunking for the history Wikipedia corpus.

This script:
1. Loads the filtered history corpus from JSON
2. Normalizes article text
3. Applies fixed-size token chunking with overlap using tiktoken
4. Saves chunked output to a new JSON file

Baseline configuration:
- No section-aware chunking
- No title injection into chunk text
- No special document structure handling
- Token counting via tiktoken cl100k_base encoding
"""

import json
import os
import re
from typing import Dict, List

import tiktoken


INPUT_PATH = "data/raw/history_corpus_clean.json"
OUTPUT_PATH = "data/chunks/chunks_size_256.json"

CHUNK_SIZE = 256
OVERLAP = 25


def normalize_text(text: str) -> str:
    """
    Cleans whitespace and removes excessive blank lines from raw article text.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line and collapse repeated blank lines
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    previous_empty = False
    for line in lines:
        is_empty = (line == "")
        if is_empty:
            if not previous_empty:
                cleaned_lines.append("")
            previous_empty = True
        else:
            cleaned_lines.append(line)
            previous_empty = False

    text = "\n".join(cleaned_lines).strip()

    # Collapse excessive internal spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text


def create_chunks(
    documents: List[Dict],
    encoder: tiktoken.Encoding,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """
    Tokenizes each document and splits it into overlapping fixed-size chunks.
    Chunk text is decoded back from tokens so it remains readable.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    all_chunks: List[Dict] = []
    step = chunk_size - overlap

    for doc in documents:
        doc_id = str(doc.get("id", ""))
        title = doc.get("title", "").strip()
        url = doc.get("url", "").strip()
        source_category = doc.get("source_category", "").strip()
        raw_text = doc.get("text", "")

        cleaned_text = normalize_text(raw_text)
        if not cleaned_text:
            continue

        # Encode full article text into tokens
        tokens = encoder.encode(cleaned_text)

        # Slide a window of chunk_size tokens with step = chunk_size - overlap
        start = 0
        chunk_index = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]

            # Decode tokens back to text
            chunk_text = encoder.decode(chunk_tokens).strip()

            if chunk_text:
                all_chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_index:04d}",
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "source_category": source_category,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                })
                chunk_index += 1

            start += step

    return all_chunks


def main() -> None:
    print("Loading corpus...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded documents : {len(documents)}")
    print(f"Chunk size       : {CHUNK_SIZE} tokens")
    print(f"Overlap          : {OVERLAP} tokens")

    # Load tiktoken encoder — cl100k_base is used by most modern embedding models
    encoder = tiktoken.get_encoding("cl100k_base")

    chunks = create_chunks(
        documents=documents,
        encoder=encoder,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("Chunking complete.")
    print(f"Total chunks     : {len(chunks)}")
    print(f"Avg per document : {len(chunks) / len(documents):.1f}")
    print(f"Output path      : {OUTPUT_PATH}")
    print("=" * 50)

    # Preview first 3 chunks
    print("\nFirst 3 chunks preview:")
    for chunk in chunks[:3]:
        preview = chunk["text"][:200].replace("\n", " ")
        print(
            f"  [{chunk['chunk_id']}] {chunk['title']} "
            f"| {chunk['token_count']} tokens | {preview}..."
        )


if __name__ == "__main__":
    main()