import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "Datasets"
model = SentenceTransformer('all-MiniLM-L6-v2')

def main():
    reviews_df = pd.read_pickle(PATH / "cleaned_reviews.pkl")
    review_texts = (
        reviews_df
        .dropna(subset=["text"])
        .groupby("item_id")["text"]
        .apply(lambda texts: "\n\n".join(texts.astype(str)))
        .reset_index(name="review_embedding_text")
    )
    embedding_vectors = model.encode(
        review_texts["review_embedding_text"].tolist(),
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )
    review_texts["embedding_vector"] = list(embedding_vectors)
    review_texts.to_pickle(PATH / "anime_review_embeddings.pkl")

if __name__ == "__main__":
    main()