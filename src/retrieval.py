"""
Retrieves similar historically-resolved threads to ground the reply draft.

Deliberately simple: TF-IDF cosine similarity over (customer_tweet), filtered
to the same predicted intent when possible. This is a baseline-grade choice on
purpose -- see decision_log.md for why we didn't reach for embeddings first.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:
    def __init__(self, historical_csv: str):
        self.df = pd.read_csv(historical_csv).dropna(subset=["customer_tweet"])
        # Only keep threads that actually have a historical reply to ground on
        self.df = self.df[self.df["historical_agent_reply"].fillna("").str.len() > 0].reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["customer_tweet"])

    def retrieve(self, tweet: str, intent: str = None, k: int = 3) -> list[dict]:
        pool = self.df
        pool_matrix = self.matrix
        if intent is not None and (pool["intent"] == intent).any():
            mask = (pool["intent"] == intent).values
            pool = pool[mask].reset_index(drop=True)
            pool_matrix = self.matrix[mask]

        if len(pool) == 0:
            return []

        query_vec = self.vectorizer.transform([tweet])
        sims = cosine_similarity(query_vec, pool_matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]

        return [
            {
                "customer_tweet": pool.iloc[i]["customer_tweet"],
                "historical_agent_reply": pool.iloc[i]["historical_agent_reply"],
                "similarity": float(sims[i]),
            }
            for i in top_idx
            if sims[i] > 0  # drop zero-similarity noise
        ]
