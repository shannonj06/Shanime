"""Combine the CSV baseline and the API crawl into the canonical review table.

Reads reviews_csv.pkl (from clean_reviews.py) and reviews_api.pkl (from
fetch_reviews_api.py, whichever exist) and writes cleaned_reviews.pkl, which
is what the profile-building step consumes.
"""

from pathlib import Path
import pandas as pd

path = Path(__file__).parent.parent / "Datasets"


def main():
    frames = []
    for name in ("reviews_csv.pkl", "reviews_api.pkl"):
        p = path / name
        if p.exists():
            df = pd.read_pickle(p)
            print(f"{name}: {len(df)} reviews, {df['item_id'].nunique()} anime")
            frames.append(df)
        else:
            print(f"{name}: (missing, skipped)")

    if not frames:
        print("Nothing to merge.")
        return

    merged = pd.concat(frames, ignore_index=True)
    # a title covered by both sources: keep both unless the same review shows up
    merged = merged.drop_duplicates(subset=["item_id", "source", "review_id"])

    out_path = path / "cleaned_reviews.pkl"
    merged.to_pickle(out_path)
    print(f"Merged -> {len(merged)} reviews across "
          f"{merged['item_id'].nunique()} anime -> {out_path}")


if __name__ == "__main__":
    main()
