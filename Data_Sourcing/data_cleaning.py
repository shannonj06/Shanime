import pandas as pd
from pathlib import Path
import re #this helps decode regex
import html

path = Path(__file__).parent.parent / "Datasets"
anime = pd.read_csv(path / "anime_dataset.csv")
manga = pd.read_csv(path / "manga_dataset.csv")

print(anime.keys())
print(manga.keys())
'''Index(['mal_id', 'title', 'title_english', 'title_japanese', 'type', 'source',
       'episodes', 'status', 'airing', 'aired_from', 'aired_to', 'duration',
       'rating', 'score', 'scored_by', 'rank', 'popularity', 'members',
       'favorites', 'season', 'year', 'studios', 'producers', 'licensors',
       'genres', 'themes', 'demographics', 'synopsis', 'image_url'],
      dtype='str')

Index(['mal_id', 'title', 'title_english', 'title_japanese', 'type',
       'chapters', 'volumes', 'status', 'publishing', 'published_from',
       'published_to', 'score', 'scored_by', 'rank', 'popularity', 'members',
       'favorites', 'authors', 'serializations', 'genres', 'themes',
       'demographics', 'synopsis', 'image_url'],
      dtype='str')'''

#cleaning anime dataset
anime = anime.drop(columns=["title_japanese", "airing", "licensors"])
anime["media_type"] = "anime"
manga = manga.drop(columns=["title_japanese", "publishing", "serializations"])
manga["media_type"] = "manga"
#they both share the same mal id so that's why im doing this
anime["item_id"] = "anime_" + anime["mal_id"].astype(str)
manga["item_id"] = "manga_" + manga["mal_id"].astype(str)


#text = re.sub(pattern, replacement, text) #this is how you use regex to replace text
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = html.unescape(str(text))  # Decode HTML entities
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove any remaining HTML tags
    text = re.sub(r"\[Written by MAL Rewrite\]", "", text)
    text = re.sub(r"\(Source:.*?\)", "", text)
    text = re.sub(r"\[Source:.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

anime["synopsis"] = anime["synopsis"].apply(clean_text)
manga["synopsis"] = manga["synopsis"].apply(clean_text)

def clean_lists(text):
    if pd.isnull(text):
        return []
    result = []
    str_val = str(text)
    pieces = str_val.split("|")
    for i in pieces:
        cleaned = i.strip()
        result.append(cleaned)
    return result

anime["genres"] = anime["genres"].apply(clean_lists)
anime["themes"] = anime["themes"].apply(clean_lists)
anime["demographics"] = anime["demographics"].apply(clean_lists)
manga["genres"] = manga["genres"].apply(clean_lists)
manga["themes"] = manga["themes"].apply(clean_lists)
manga["demographics"] = manga["demographics"].apply(clean_lists)

anime = anime.drop_duplicates(subset=["item_id"])
manga = manga.drop_duplicates(subset=["item_id"])

anime["norm_title"] = anime["title"].str.lower().str.strip()
manga["norm_title"] = manga["title"].str.lower().str.strip()

anime = anime.drop_duplicates(subset=["norm_title", "type"], keep="first")
manga = manga.drop_duplicates(subset=["norm_title", "type"], keep="first")

def clean_numeric_column(column):
    column = pd.to_numeric(column, errors='coerce')  # Convert to numeric, invalid parses become NaN
    return column

anime_numeric_columns = ["episodes", "score", "scored_by", "rank", "popularity", "members", "favorites"]
manga_numeric_columns = ["chapters", "volumes", "score", "scored_by", "rank", "popularity", "members", "favorites"]

for col in anime_numeric_columns:
    anime[col] = clean_numeric_column(anime[col])

for col in manga_numeric_columns:
    manga[col] = clean_numeric_column(manga[col])

anime["creators"] = anime["studios"].apply(clean_lists)
manga["creators"] = manga["authors"].apply(clean_lists)

# manga has no `year` column; derive it from published_from. For anime, fall back
# to aired_from where the existing year is missing.
anime["year"] = anime["year"].fillna(
    pd.to_datetime(anime["aired_from"], errors="coerce").dt.year
)
manga["year"] = pd.to_datetime(manga["published_from"], errors="coerce").dt.year

def join_list(items):
    if isinstance(items, list):
        return ", ".join(items)
    return ""

def has_embedding_info(row):
    has_title = isinstance(row["title"], str) and row["title"].strip() != ""
    has_synopsis = len(row["synopsis"]) >= 20
    has_genres = len(row["genres"]) > 0
    has_themes = len(row["themes"]) > 0
    has_demo = len(row["demographics"]) > 0
    
    return has_title and (has_synopsis or has_genres or has_themes or has_demo)

def format_year(year):
    if pd.isnull(year):
        return ""
    return str(int(year))

def make_embedding_text(row):
    parts = [
        f"Title: {row.get('title', '')}",
        f"English title: {row.get('title_english', '')}",
        f"Media type: {row.get('media_type', '')}",
        f"Format: {row.get('type', '')}",
        f"Genres: {join_list(row.get('genres', []))}",
        f"Themes: {join_list(row.get('themes', []))}",
        f"Demographics: {join_list(row.get('demographics', []))}",
        f"Creators: {join_list(row.get('creators', []))}",
        f"Year: {format_year(row.get('year'))}",
        f"Synopsis: {row.get('synopsis', '')}"
    ]
    return "\n".join([p for p in parts if p and not p.endswith(": ")])

# drop items with no usable text/metadata to embed
anime = anime[anime.apply(has_embedding_info, axis=1)].copy()
manga = manga[manga.apply(has_embedding_info, axis=1)].copy()

anime["embedding_text"] = anime.apply(make_embedding_text, axis=1)
manga["embedding_text"] = manga.apply(make_embedding_text, axis=1)

final_cols = [
    "item_id",
    "mal_id",
    "media_type",
    "title",
    "title_english",
    "type",
    "status",
    "year",
    "genres",
    "themes",
    "demographics",
    "creators",
    "synopsis",
    "score",
    "scored_by",
    "rank",
    "popularity",
    "members",
    "favorites",
    "image_url",
    "embedding_text"
]

items = pd.concat(
    [anime[final_cols], manga[final_cols]],
    ignore_index=True
)

# persist cleaned output for the embedding step. pickle keeps the list columns
# (genres/themes/etc.) intact with no extra dependencies; read back with
# pd.read_pickle(out_path).
out_path = path / "cleaned_items.pkl"
items.to_pickle(out_path)
print(f"Saved {len(items)} items to {out_path}")
