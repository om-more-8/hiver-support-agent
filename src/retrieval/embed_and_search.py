"""
Embeds all historical (customer_text -> brand_reply) pairs once, then
retrieves the top-k most similar past conversations for a new incoming
message. This is what "grounds" reply generation in real historical
resolutions instead of the LLM free-generating.

USAGE:
    from src.retrieval.embed_and_search import Retriever
    r = Retriever()
    r.search("my package is late", k=3)
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import PROCESSED_DIR, BRAND, EMBEDDING_MODEL, TOP_K_RETRIEVAL


class Retriever:
    def __init__(self, brand: str = BRAND, rebuild: bool = False):
        self.pairs_path = PROCESSED_DIR / f"{brand}_pairs.csv"
        self.cache_path = PROCESSED_DIR / f"{brand}_embeddings.pkl"
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.pairs = pd.read_csv(self.pairs_path)

        if self.cache_path.exists() and not rebuild:
            with open(self.cache_path, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            print(f"Embedding {len(self.pairs)} customer messages (one-time cost, cached after)...")
            self.embeddings = self.model.encode(
                self.pairs["customer_text"].astype(str).tolist(),
                show_progress_bar=True, batch_size=64,
            )
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.embeddings, f)

    def search(self, query_text: str, k: int = TOP_K_RETRIEVAL) -> list[dict]:
        query_emb = self.model.encode([query_text])[0]
        sims = self.embeddings @ query_emb / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            results.append({
                "customer_text": self.pairs.iloc[idx]["customer_text"],
                "brand_reply_text": self.pairs.iloc[idx]["brand_reply_text"],
                "similarity": float(sims[idx]),
            })
        return results


if __name__ == "__main__":
    r = Retriever()
    results = r.search("my order still hasn't arrived, its been 2 weeks")
    for res in results:
        print(f"\nsim={res['similarity']:.3f}")
        print(f"  customer: {res['customer_text'][:100]}")
        print(f"  reply:    {res['brand_reply_text'][:100]}")