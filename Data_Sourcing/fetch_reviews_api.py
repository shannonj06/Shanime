"""Fetch anime reviews from the Jikan API to fill the gap the CSV can't cover.

The static reviews.csv only covers ~8k anime. This crawls the rest via the API.
Because Jikan is rate-limited (~60 req/min), a full crawl of the remaining
~22k anime takes several hours, so this is built to be RESUMABLE:

  * progress is written incrementally to a JSONL cache (one line per review)
  * every attempted item_id (even ones with zero reviews) is recorded in a
    done-set file, so a re-run skips them instead of re-fetching
  * titles are visited most-popular-first, so an interrupted run still covers
    the anime most likely to be recommended

Usage:
    python fetch_reviews_api.py --limit 30     # small test batch
    python fetch_reviews_api.py                # full gap crawl (hours)
    python fetch_reviews_api.py --all          # re-fetch every anime, incl. CSV-covered
    python fetch_reviews_api.py --finalize     # just rebuild reviews_api.pkl from cache

Then run merge_reviews.py to combine CSV + API into cleaned_reviews.pkl.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from review_sourcing import fetch_all_reviews
from clean_reviews import clean_review_text

path = Path(__file__).parent.parent / "Datasets"
CACHE_JSONL = path / "api_reviews_cache.jsonl"   # raw fetched reviews, appended
DONE_IDS = path / "api_fetched_ids.txt"          # item_ids already attempted


def load_done_ids():
    if DONE_IDS.exists():
        return set(DONE_IDS.read_text(encoding="utf-8").split())
    return set()


def target_anime(fetch_all):
    """Return anime item_ids to fetch, most-popular-first."""
    items = pd.read_pickle(path / "cleaned_items.pkl")
    anime = items[items["media_type"] == "anime"].copy()

    if not fetch_all:
        csv_path = path / "reviews_csv.pkl"
        if csv_path.exists():
            covered = set(pd.read_pickle(csv_path)["item_id"])
            anime = anime[~anime["item_id"].isin(covered)]

    # most members first -> important titles covered even if interrupted
    anime = anime.sort_values("members", ascending=False, na_position="last")
    return anime[["item_id", "mal_id", "title"]].reset_index(drop=True)


def crawl(limit=None, fetch_all=False, max_pages=2):
    todo = target_anime(fetch_all)
    done = load_done_ids()
    todo = todo[~todo["item_id"].isin(done)]
    if limit:
        todo = todo.head(limit)

    print(f"{len(done)} already fetched; crawling {len(todo)} anime "
          f"(max_pages={max_pages})")

    n_reviews = 0
    with open(CACHE_JSONL, "a", encoding="utf-8") as cache, \
         open(DONE_IDS, "a", encoding="utf-8") as donef:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            try:
                reviews = fetch_all_reviews(row.mal_id, "anime", max_pages=max_pages)
            except Exception as e:  # never let one title kill the crawl
                print(f"  ! {row.item_id} ({row.title}): {e}")
                continue
            for r in reviews:
                r["item_id"] = row.item_id
                cache.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_reviews += len(reviews)
            donef.write(row.item_id + "\n")
            # flush so an interrupt loses at most the current title
            cache.flush()
            donef.flush()
            if i % 25 == 0:
                print(f"  [{i}/{len(todo)}] {row.title[:40]!r:42} "
                      f"+{len(reviews)} reviews (total {n_reviews})")
    print(f"Done. Wrote {n_reviews} reviews this run.")
    finalize()


def finalize():
    """Rebuild reviews_api.pkl from the JSONL cache (cleaned + normalized)."""
    if not CACHE_JSONL.exists():
        print("No cache yet; nothing to finalize.")
        return
    rows = []
    with open(CACHE_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("Cache empty.")
        return
    df = pd.DataFrame(rows)
    df["text"] = df["text"].apply(clean_review_text)
    df = df[df["text"].str.len() >= 40]
    df = df.drop_duplicates(subset=["item_id", "review_id"])
    df["source"] = "api"
    df = df[["item_id", "review_id", "profile", "score", "text", "source"]]

    out_path = path / "reviews_api.pkl"
    df.to_pickle(out_path)
    print(f"Finalized {len(df)} API reviews "
          f"({df['item_id'].nunique()} anime) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch this many anime (for testing)")
    ap.add_argument("--all", action="store_true",
                    help="fetch every anime, including CSV-covered ones")
    ap.add_argument("--max-pages", type=int, default=2,
                    help="pages of reviews per title (20 each)")
    ap.add_argument("--finalize", action="store_true",
                    help="just rebuild reviews_api.pkl from cache and exit")
    args = ap.parse_args()

    if args.finalize:
        finalize()
    else:
        crawl(limit=args.limit, fetch_all=args.all, max_pages=args.max_pages)
