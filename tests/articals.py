import json
from collections import Counter
import pandas as pd

CORPUS_PATH = "data/raw/history_corpus_clean.json"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Ukupan broj članaka
total_articles = len(corpus)

print(f"Ukupan broj članaka: {total_articles}")

# Broj članaka po kategoriji
category_counts = Counter(article.get("source_category", "UNKNOWN") for article in corpus)

category_df = (
    pd.DataFrame(category_counts.items(), columns=["source_category", "num_articles"])
    .sort_values("num_articles", ascending=False)
    .reset_index(drop=True)
)

print("\nBroj članaka po kategoriji:")
print(category_df)

# Osnovna statistika dužine teksta
lengths = []

for article in corpus:
    text = article.get("text", "")
    text_length = article.get("text_length", len(text))

    lengths.append({
        "id": article.get("id"),
        "title": article.get("title"),
        "source_category": article.get("source_category", "UNKNOWN"),
        "text_length": text_length
    })

length_df = pd.DataFrame(lengths)

print("\nStatistika dužine članaka:")
print(length_df["text_length"].describe())

print("\nProsečna dužina teksta po kategoriji:")
avg_length_by_category = (
    length_df
    .groupby("source_category")["text_length"]
    .agg(
        num_articles="count",
        avg_text_length="mean",
        median_text_length="median",
        min_text_length="min",
        max_text_length="max"
    )
    .sort_values("num_articles", ascending=False)
    .reset_index()
)

print(avg_length_by_category)

# Najkraći i najduži članci
print("\nNajkraćih 10 članaka:")
print(
    length_df
    .sort_values("text_length", ascending=True)
    [["title", "source_category", "text_length"]]
    .head(10)
)

print("\nNajdužih 10 članaka:")
print(
    length_df
    .sort_values("text_length", ascending=False)
    [["title", "source_category", "text_length"]]
    .head(10)
)