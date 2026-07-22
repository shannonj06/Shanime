from pathlib import Path

from agents import function_tool
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PATH = Path(__file__).resolve().parent.parent / "Datasets"

# reset_index so row labels equal positional indices into the score arrays below
meta_df = pd.read_pickle(PATH / "meta_embeddings.pkl").reset_index(drop=True)
review_df = pd.read_pickle(PATH / "anime_review_embeddings.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- build once, aligned to meta_df's row order by item_id ---
meta_matrix = np.vstack(meta_df["embedding_vector"].to_numpy())
dim = meta_matrix.shape[1]

review_lookup = dict(zip(review_df["item_id"], review_df["embedding_vector"]))
review_matrix = np.vstack(
    [review_lookup.get(iid, np.zeros(dim)) for iid in meta_df["item_id"]]
)
has_review = np.array([iid in review_lookup for iid in meta_df["item_id"]])


def _list_values(column):
    return sorted(
        {
            value
            for lst in meta_df[column].dropna()
            if isinstance(lst, list)
            for value in lst
            if value
        }
    )


min_year = int(meta_df["year"].dropna().min())
max_year = int(meta_df["year"].dropna().max())

# Two ways to talk about popularity, and they run in opposite directions:
#   members    = how many users have the show on a list. Bigger = more popular.
#   popularity = MAL's popularity *rank*. 1 = most popular, so SMALLER = more popular.
# The rank filters are named popularity_rank_* so callers can't read them as a
# magnitude and invert the meaning by accident. Rank is also scoped per media_type
# (anime and manga each have their own #1), so it only means anything alongside a
# media_type filter -- search() sets one by default. members is a raw count and is
# comparable across both.
min_members = int(meta_df["members"].dropna().min())
max_members = int(meta_df["members"].dropna().max())
min_popularity_rank = int(meta_df["popularity"].dropna().min())
max_popularity_rank = int(meta_df["popularity"].dropna().max())

FILTERS = {
    "media_type": {"column": "media_type", "type": "category", "allowed": ["anime", "manga"]},
    "genres": {"column": "genres", "type": "list_contains", "allowed": _list_values("genres")},
    "themes": {"column": "themes", "type": "list_contains", "allowed": _list_values("themes")},
    "year_min": {"column": "year", "type": "min", "allowed": [min_year]},
    "year_max": {"column": "year", "type": "max", "allowed": [max_year]},
    "score_min": {"column": "score", "type": "min", "allowed": [0]},
    "score_max": {"column": "score", "type": "max", "allowed": [10]},
    # how many people actually rated it -- a high score off 100 votes is noise, so
    # this is the guard that makes score_min mean something on obscure titles
    "scored_by_min": {"column": "scored_by", "type": "min", "allowed": [0]},
    # audience size: members_min = "well known", members_max = "obscure"
    "members_min": {"column": "members", "type": "min", "allowed": [min_members]},
    "members_max": {"column": "members", "type": "max", "allowed": [max_members]},
    # rank: popularity_rank_max = "inside the top N most popular" (the usual ask),
    # popularity_rank_min = "ranked no higher than N", i.e. deliberately off the radar
    "popularity_rank_min": {
        "column": "popularity", "type": "min", "allowed": [min_popularity_rank]
    },
    "popularity_rank_max": {
        "column": "popularity", "type": "max", "allowed": [max_popularity_rank]
    },
}


def apply_filters(frame, filters):
    for key, value in filters.items():
        if value is None or value == "" or key not in FILTERS:
            continue
        f = FILTERS[key]
        col = f["column"]
        if f["type"] == "category":
            frame = frame[frame[col] == value]
        elif f["type"] == "list_contains":
            # guard against NaN / non-list cells so `value in lst` can't raise
            frame = frame[frame[col].apply(lambda lst: isinstance(lst, list) and value in lst)]
        elif f["type"] == "min":
            frame = frame[frame[col] >= value]
        elif f["type"] == "max":
            frame = frame[frame[col] <= value]
    return frame

def search(query, top_k=5, meta_weight=.7, review_weight=.3, filters=None, media_type="anime"):
    # Default to anime-only; pass media_type=None (and no filter) to include manga.
    filters = dict(filters) if filters else {}
    if media_type is not None:
        filters.setdefault("media_type", media_type)

    candidates = apply_filters(meta_df, filters)
    if candidates.empty:
        return candidates

    q = model.encode(query, normalize_embeddings=True)
    meta_scores = meta_matrix @ q
    review_scores = review_matrix @ q
    # items without reviews: use meta score alone (don't penalize them with a 0)
    total = np.where(
        has_review,
        meta_weight * meta_scores + review_weight * review_scores,
        meta_scores,
    )

    positions = candidates.index.to_numpy()
    order = np.argsort(total[positions])[-top_k:][::-1]
    top = positions[order]

    results = meta_df.iloc[top].copy()
    results["search_score"] = total[top]
    return results


# Python callers use search(); agents that need to search on their own use this.
search_tool = function_tool(search)
