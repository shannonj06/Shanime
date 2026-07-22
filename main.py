import pandas as pd
from pathlib import Path
from Data_Sourcing.review_sourcing import fetch_reviews

PATH = Path(__file__).resolve().parent / "Datasets"
BASE_URL = "https://api.jikan.moe/v4"
params = {
    "page": 1,
    "preliminary": "true",
    "spoilers": "true"
}

def main():
    df =pd.read_pickle(PATH / "cleaned_items.pkl")
    print("Data loaded successfully.")
    print(df.keys())
    '''Index(['item_id', 'mal_id', 'media_type', 'title', 'title_english', 'type',
       'status', 'year', 'genres', 'themes', 'demographics', 'creators',
       'synopsis', 'score', 'scored_by', 'rank', 'popularity', 'members',
       'favorites', 'image_url', 'embedding_text'],
      dtype='str')'''
    
    reviews = fetch_reviews(BASE_URL, params=params, value=1, media_type="anime")
    print(reviews['data'])
    

if __name__ == "__main__":
    main()