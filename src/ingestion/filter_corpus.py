"""
Corpus Filter - Filters history corpus articles based on quality and relevance criteria.
"""

import json
from collections import Counter
from typing import Dict, List

INPUT_PATH = "data/raw/history_corpus.json"
OUTPUT_PATH = "data/raw/history_corpus_clean.json"

REMOVE_CATEGORIES = {
    "Byzantine Empire",
    "Middle Ages",
    "History of the United States",
    "World War II by country",
}

TITLE_BLOCKLIST = [
    "bibliography", "historiography", "list of", "timeline of",
    "outline of", "index of", "glossary", "disambiguation",
    " births", " deaths", "in history", "category:", "portal:",
    "template:", "file:", "wikipedia:", "help:",
    "age of revolution", "batavian revolution", "allied sovereigns",
    "calendar", "coin", "coinage", "daric", "danake",
    "siglos", "postal", "chiliarch",
]

TOPIC_BLOCKLIST = [
    "divination", "etymology", "philology", "dance",
 "linguistics", "grammar", "comedy",
]

CORE_HISTORY_KEYWORDS = [
    "war", "battle", "army", "empire", "king", "conquest",
    "reign", "dynasty", "military", "campaign", "treaty",
]

MIN_CORE_HITS = 3
MIN_TEXT_LENGTH = 3000
MIN_KEYWORD_HITS = 5


def passes_filters(article: Dict) -> bool:
    title = article.get("title", "").strip().lower()
    text = article.get("text", "").strip().lower()

    if not title or not text:
        return False

    if len(text) < MIN_TEXT_LENGTH:
        return False

    for blocked in TITLE_BLOCKLIST:
        if blocked in title:
            return False

    for blocked in TOPIC_BLOCKLIST:
        if blocked in text:
            return False

    keyword_hits = sum(1 for kw in CORE_HISTORY_KEYWORDS if kw in text)
    if keyword_hits < MIN_CORE_HITS:
        return False

    return True


def filter_corpus(input_path: str, output_path: str) -> List[Dict]:
    with open(input_path, encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Before filtering: {len(articles)} articles")

    # Remove unwanted categories
    articles = [a for a in articles if a.get("source_category") not in REMOVE_CATEGORIES]
    print(f"After category removal: {len(articles)} articles")

    # Apply quality and relevance filters
    kept = [a for a in articles if passes_filters(a)]

    print(f"After content filtering: {len(kept)} articles")
    print(f"\nRemoved categories: {REMOVE_CATEGORIES}")
    print(f"\nFinal distribution:")
    counts = Counter(a["source_category"] for a in kept)
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<35} {count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    return kept


if __name__ == "__main__":
    filter_corpus(INPUT_PATH, OUTPUT_PATH)