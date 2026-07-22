import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "Datasets"
model = SentenceTransformer('all-MiniLM-L6-v2')

def main():
    meta_df = pd.read_pickle(PATH / "cleaned_items.pkl")
    texts = meta_df["embedding_text"].fillna("").astype(str).tolist()
    embedding_vectors = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )
    meta_df["embedding_vector"] = list(embedding_vectors)
    meta_df.to_pickle(PATH / "meta_embeddings.pkl")

if __name__ == "__main__":
    main()

