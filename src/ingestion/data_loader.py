"""
Data Loader - Downloads a domain-limited Wikipedia history corpus via Wikipedia API
and saves it locally for further processing.

Categories cover ancient, medieval, early modern, and modern history
to ensure a diverse and balanced corpus for RAG experimentation.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import requests
from tqdm import tqdm


DEFAULT_OUTPUT_PATH = "data/raw/history_corpus.json"

HEADERS = {
    "User-Agent": "RAGMasterThesis/1.0 (academic research project)"
}

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

HISTORY_CATEGORIES = [
    # Ancient history
    "Ancient history",
    "Ancient Egypt",
    "Ancient Rome",
    "Ancient Greece",
    "Achaemenid Empire",
    # Medieval
    "Crusades",
    # Early modern
    "Napoleonic Wars",
    "Ottoman Empire",
    # Modern
    "World War I",
    "World War II",
    # Regional
    "History of Europe",
    "American Civil War",
    "American Revolution",
]

ARTICLES_PER_CATEGORY = 100
MIN_TEXT_LENGTH = 3000
REQUEST_DELAY_SECONDS = 1
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 10
MIN_KEYWORD_HITS = 5

TITLE_BLOCKLIST = [
    "bibliography", "historiography", "list of", "timeline of",
    "outline of", "index of", "glossary", "disambiguation",
    " births", " deaths", "in history", "category:", "portal:",
    "template:", "file:", "wikipedia:", "help:",
]

HISTORY_KEYWORDS = [
    # Classical history
    "empire", "dynasty", "war", "battle", "king", "queen", "pharaoh",
    "republic", "army", "treaty", "conquest", "revolt", "civilization",
    "century", "ancient", "medieval", "reign", "throne", "kingdom",
    "emperor", "general", "colony", "trade", "religion", "temple",
    "aristocracy", "senate", "parliament", "revolution", "crusade",

    # Modern warfare — WW2
    "nazi", "fascist", "holocaust", "occupation", "resistance",
    "allied", "axis", "infantry", "offensive", "armistice",
    "blitzkrieg", "liberation", "deportation", "propaganda",
    "luftwaffe", "wehrmacht", "reich",

    # US History
    "congress", "president", "constitution", "confederation",
    "colonial", "independence", "slavery", "plantation",
    "reconstruction", "founding fathers", "amendment",
    "union", "confederacy", "abolitionist",

    # Medieval
    "feudal", "papal", "monastery", "knight", "plague",
    "cathedral", "serfdom", "bishop", "excommunication",
    "vassal", "crusader", "pilgrimage", "inquisition",

    # Byzantine
    "byzantine", "constantinople", "orthodox", "patriarch",
    "themata", "basileus", "iconoclasm",
]


def passes_quality_filters(title: str, text: str) -> bool:
    """
    Returns True if the article passes all quality and relevance filters:
    title blocklist, year-pattern rejection, minimum length, keyword relevance,
    list/table ratio, and minimum prose sentence count.
    """
    title = title.strip()
    text = text.strip()

    if not title or not text:
        return False

    title_lower = title.lower()
    text_lower = text.lower()

    for blocked in TITLE_BLOCKLIST:
        if blocked in title_lower:
            return False

    if title_lower.isdigit():
        return False
    if re.fullmatch(r"\d{3,4}", title_lower):
        return False
    if re.match(r"^\d{3,4} in ", title_lower):
        return False

    if len(text) < MIN_TEXT_LENGTH:
        return False

    keyword_hits = sum(1 for kw in HISTORY_KEYWORDS if kw in text_lower)
    if keyword_hits < MIN_KEYWORD_HITS:
        return False

    lines = text.splitlines()
    if not lines:
        return False
    list_lines = sum(1 for line in lines if line.strip().startswith(("*", "#", "|", "!")))
    if list_lines / len(lines) > 0.40:
        return False

    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 40]
    if len(sentences) < 10:
        return False

    return True


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def safe_get_json(
    session: requests.Session,
    url: str,
    params: Dict,
    timeout: int = 20,
) -> Optional[Dict]:
    """
    Performs a GET request with retry logic and exponential backoff on 429 responses.
    Returns the parsed JSON response, or None if all retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=timeout)

            if response.status_code == 429:
                wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
                print(f"\n  [rate limit] waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\n  [error] {e}")
                return None
            time.sleep(1.0)

    return None


def get_category_articles(
    session: requests.Session,
    category: str,
    limit: int = 100,
) -> List[str]:
    """
    Returns up to `limit` article titles from a Wikipedia category,
    using pagination to go beyond the API's single-request maximum.
    """
    titles: List[str] = []
    cmcontinue: Optional[str] = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(100, limit - len(titles)),
            "cmtype": "page",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = safe_get_json(session, WIKIPEDIA_API_URL, params)
        if not data:
            break

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend([m["title"] for m in members if "title" in m])

        if "continue" not in data or "cmcontinue" not in data["continue"]:
            break

        cmcontinue = data["continue"]["cmcontinue"]
        time.sleep(REQUEST_DELAY_SECONDS)

    return titles[:limit]


def get_article_text(session: requests.Session, title: str) -> Optional[Dict]:
    """
    Fetches the plain-text extract of a Wikipedia article by title.
    Returns a metadata dict or None if the article is missing or fails quality filters.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }

    data = safe_get_json(session, WIKIPEDIA_API_URL, params)
    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        return None

    title_clean = page.get("title", title).strip()
    text = page.get("extract", "").strip()

    if not passes_quality_filters(title_clean, text):
        return None

    return {
        "id": str(page.get("pageid")),
        "title": title_clean,
        "text": text,
        "url": f"https://en.wikipedia.org/wiki/{quote(title_clean.replace(' ', '_'))}",
        "text_length": len(text),
    }


def load_history_corpus(
    output_path: str = DEFAULT_OUTPUT_PATH,
    articles_per_category: int = ARTICLES_PER_CATEGORY,
    max_total_articles: Optional[int] = None,
) -> List[Dict]:
    """
    Downloads Wikipedia history articles across all configured categories
    and saves the result as a JSON file.
    """
    print("Loading Wikipedia history corpus...")
    print(f"Categories       : {len(HISTORY_CATEGORIES)}")
    print(f"Per category cap : {articles_per_category}")
    print(f"Min text length  : {MIN_TEXT_LENGTH} chars")
    print(f"Min keyword hits : {MIN_KEYWORD_HITS}")

    session = make_session()
    print("Waiting 60s before starting to let API cool down...")
    time.sleep(60)

    all_articles: List[Dict] = []
    seen_titles: Set[str] = set()
    total_fetched = 0
    total_rejected = 0

    for category in HISTORY_CATEGORIES:
        print(f"\nFetching category: {category}")

        titles = get_category_articles(session, category, limit=articles_per_category)
        print(f"  Found {len(titles)} titles in category")

        category_articles: List[Dict] = []

        for title in tqdm(titles, desc="  Downloading"):
            normalized_title = title.strip().lower()
            if normalized_title in seen_titles:
                continue

            total_fetched += 1
            article = get_article_text(session, title)

            if article:
                article["source_category"] = category
                category_articles.append(article)
                seen_titles.add(normalized_title)

                if (
                    max_total_articles is not None
                    and len(all_articles) + len(category_articles) >= max_total_articles
                ):
                    break
            else:
                total_rejected += 1

            time.sleep(REQUEST_DELAY_SECONDS)

        print(f"  Accepted {len(category_articles)} / {len(titles)} articles")
        all_articles.extend(category_articles)

        # Save checkpoint after each category
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        if max_total_articles is not None and len(all_articles) >= max_total_articles:
            print(f"\nReached max_total_articles limit ({max_total_articles}). Stopping.")
            break

        time.sleep(5.0)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    acceptance_rate = (
        round((total_fetched - total_rejected) / total_fetched * 100, 1)
        if total_fetched > 0 else 0
    )

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Total fetched   : {total_fetched}")
    print(f"  Total rejected  : {total_rejected}")
    print(f"  Acceptance rate : {acceptance_rate}%")
    print(f"  Saved articles  : {len(all_articles)}")
    print(f"  Output path     : {output_path}")
    print(f"{'='*50}")

    return all_articles


if __name__ == "__main__":
    articles = load_history_corpus(
        output_path=DEFAULT_OUTPUT_PATH,
        articles_per_category=100,
        max_total_articles=1000,
    )

    if articles:
        print("\nFirst 10 article titles:")
        for a in articles[:10]:
            print(f"  - {a['title']} ({a['source_category']}, {a['text_length']} chars)")
    else:
        print("No articles found.")