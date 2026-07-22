"""Step 1 of the emotional-recommender pipeline.

Take the raw Kaggle-style `reviews.csv` (anime only), strip the scraped
scorecard boilerplate + HTML noise, key each review to the same `item_id`
used in cleaned_items.pkl, and persist everything as `cleaned_reviews.pkl`.

We store ALL reviews (not a truncated sample): later steps distill them into
per-item emotional profiles, and keeping the raw text lets us re-distill
without re-processing. Read back with pd.read_pickle(out_path).
"""

import pandas as pd
from pathlib import Path
import re
import html

path = Path(__file__).parent.parent / "Datasets"

# scorecard labels the MAL scraper dumps at the top of every review, e.g.
# "Overall 8 Story 8 Animation 8 Sound 10 Character 9 Enjoyment 8"
SCORECARD_LABELS = ["Overall", "Story", "Animation", "Sound", "Character", "Enjoyment"]
_scorecard_re = re.compile(
    r"\b(?:" + "|".join(SCORECARD_LABELS) + r")\b\s*\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)


def clean_review_text(text):
    if pd.isnull(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)          # scraped line breaks
    text = _scorecard_re.sub(" ", text)             # strip the score block
    text = re.sub(r"\bmore pics\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Written by MAL Rewrite\]", "", text)
    text = re.sub(r"\(Source:.*?\)", "", text)
    text = re.sub(r"\[Source:.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace mess
    return text


def main():
    reviews = pd.read_csv(path / "reviews.csv")
    print(f"Loaded {len(reviews)} raw reviews for {reviews['anime_uid'].nunique()} anime")

    # match the item_id scheme from data_cleaning.py: "anime_" + mal_id
    reviews["item_id"] = "anime_" + reviews["anime_uid"].astype(str)

    # only keep reviews for items that survived cleaning (drops de-duped / thin items)
    items = pd.read_pickle(path / "cleaned_items.pkl")
    valid_ids = set(items["item_id"])
    before = len(reviews)
    reviews = reviews[reviews["item_id"].isin(valid_ids)].copy()
    print(f"Kept {len(reviews)} of {before} reviews that map to a cleaned item")

    reviews["text"] = reviews["text"].apply(clean_review_text)

    # drop reviews that are basically empty once boilerplate is stripped
    reviews = reviews[reviews["text"].str.len() >= 40].copy()

    out = reviews.rename(columns={"uid": "review_id"})[
        ["item_id", "review_id", "profile", "score", "text"]
    ].reset_index(drop=True)
    out["source"] = "csv"

    # source-specific artifact; merge_reviews.py combines it with the API pull
    out_path = path / "reviews_csv.pkl"
    out.to_pickle(out_path)
    print(f"Saved {len(out)} cleaned CSV reviews "
          f"({out['item_id'].nunique()} anime) to {out_path}")


if __name__ == "__main__":
    main()
