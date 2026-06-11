"""
Contextual chunking for the history Wikipedia corpus.

This script adds section-aware context to each chunk by prepending
the article title and current section heading as a prefix.
Bibliographic sections (references, external links, etc.) are skipped.
"""

import json
import os
import re
from typing import Dict, List

import tiktoken


INPUT_PATH = "data/raw/history_corpus_clean.json"
OUTPUT_PATH = "data/chunks/chunks_contextual.json"

CHUNK_SIZE = 512
OVERLAP = 50

SKIP_SECTIONS = [
    "see also",
    "bibliography",
    "references",
    "further reading",
    "external links",
    "notes",
    "sources",
    "citations",
    "notes and references",
    "footnotes",
    "works cited",
]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
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
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_sections(text: str) -> List[Dict]:
    """
    Split article text into sections based on == headings.
    Returns list of {heading, text} dicts.
    """
    section_pattern = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)

    sections = []
    last_end = 0
    current_heading = "Introduction"

    for match in section_pattern.finditer(text):
        section_text = text[last_end:match.start()].strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "text": section_text,
            })
        current_heading = match.group(2).strip()
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        sections.append({
            "heading": current_heading,
            "text": remaining,
        })

    return sections


def is_bibliographic_section(heading: str) -> bool:
    """Return True if the section should be skipped."""
    heading_lower = heading.lower().strip()
    return any(skip in heading_lower for skip in SKIP_SECTIONS)


def create_chunks(
    documents: List[Dict],
    encoder: tiktoken.Encoding,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    all_chunks: List[Dict] = []
    skipped_sections = 0

    for doc in documents:
        doc_id = str(doc.get("id", ""))
        title = doc.get("title", "").strip()
        url = doc.get("url", "").strip()
        source_category = doc.get("source_category", "").strip()
        raw_text = doc.get("text", "")

        cleaned_text = normalize_text(raw_text)
        if not cleaned_text:
            continue

        sections = extract_sections(cleaned_text)

        chunk_index = 0

        for section in sections:
            heading = section["heading"]
            section_text = section["text"]

            if not section_text.strip():
                continue

            # Skip bibliographic sections
            if is_bibliographic_section(heading):
                skipped_sections += 1
                continue

            # Build context prefix
            prefix = f"[{title} > {heading}]\n\n"
            prefix_tokens = encoder.encode(prefix)
            prefix_len = len(prefix_tokens)

            tokens = encoder.encode(section_text)

            effective_chunk_size = chunk_size - prefix_len
            if effective_chunk_size <= 0:
                continue

            effective_step = effective_chunk_size - overlap
            if effective_step <= 0:
                effective_step = 1

            start = 0
            while start < len(tokens):
                end = start + effective_chunk_size
                chunk_tokens = tokens[start:end]

                chunk_text = encoder.decode(chunk_tokens).strip()

                if chunk_text:
                    contextual_text = prefix + chunk_text

                    all_chunks.append({
                        "chunk_id": f"{doc_id}_{chunk_index:04d}",
                        "doc_id": doc_id,
                        "title": title,
                        "url": url,
                        "source_category": source_category,
                        "section_heading": heading,
                        "chunk_index": chunk_index,
                        "text": contextual_text,
                        "token_count": len(prefix_tokens) + len(chunk_tokens),
                    })
                    chunk_index += 1

                start += effective_step

    return all_chunks, skipped_sections


def main() -> None:
    print("Loading corpus...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded documents  : {len(documents)}")
    print(f"Chunk size        : {CHUNK_SIZE} tokens")
    print(f"Overlap           : {OVERLAP} tokens")
    print(f"Strategy          : Contextual (title + section prefix)")

    encoder = tiktoken.get_encoding("cl100k_base")

    chunks, skipped_sections = create_chunks(
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
    print(f"Total chunks      : {len(chunks)}")
    print(f"Skipped sections  : {skipped_sections}")
    print(f"Avg per document  : {len(chunks) / len(documents):.1f}")
    print(f"Output path       : {OUTPUT_PATH}")
    print("=" * 50)

    print("\nFirst 3 chunks preview:")
    for chunk in chunks[:3]:
        preview = chunk["text"][:300].replace("\n", " ")
        print(
            f"  [{chunk['chunk_id']}] {chunk['title']} > {chunk['section_heading']} "
            f"| {chunk['token_count']} tokens | {preview}..."
        )


if __name__ == "__main__":
    main()