"""
Hybrid retrieval over the policy chunks.

We keep this simple and dependency-light (no external embedding API, no internet
needed at query time) so a beginner can run it anywhere:

  - Sparse retrieval: BM25 (rank_bm25) over tokenized chunk text.
  - Dense retrieval:  TF-IDF vectors compressed with Truncated SVD ("simple
                       embeddings"). This captures topical/semantic similarity
                       beyond exact keyword overlap, without needing a
                       downloaded neural embedding model.
  - Fusion:           Reciprocal Rank Fusion (RRF) combines the two ranked
                       lists into one, so we don't need to tune a weight
                       between "sparse score" and "dense score" (they live on
                       different scales).
  - Rerank:           A lightweight keyword-overlap reranker re-scores the
                       fused top-K using the *specific* query terms, which
                       tends to push the most on-point clause to the top.
"""

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ingestion import Chunk

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    sparse_rank: int
    dense_rank: int
    fused_score: float
    rerank_score: float


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]

        # sparse index
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # dense index: TF-IDF -> SVD ("semantic" vectors)
        raw_texts = [c.text for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        tfidf_matrix = self.vectorizer.fit_transform(raw_texts)
        n_components = min(100, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_components = max(2, n_components)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dense_matrix = self.svd.fit_transform(tfidf_matrix)
        # normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.dense_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.dense_matrix_norm = self.dense_matrix / norms

    def _sparse_rank(self, query: str, top_k: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1][:top_k])

    def _dense_rank(self, query: str, top_k: int) -> List[int]:
        q_vec = self.vectorizer.transform([query])
        q_dense = self.svd.transform(q_vec)
        q_norm = np.linalg.norm(q_dense)
        if q_norm == 0:
            q_norm = 1
        q_dense = q_dense / q_norm
        sims = self.dense_matrix_norm @ q_dense.T
        sims = sims.flatten()
        return list(np.argsort(sims)[::-1][:top_k])

    def _rerank_score(self, query: str, text: str) -> float:
        """Simple, transparent keyword-overlap reranker (no extra model needed)."""
        q_terms = set(tokenize(query))
        t_terms = set(tokenize(text))
        if not q_terms:
            return 0.0
        overlap = len(q_terms & t_terms)
        return overlap / len(q_terms)

    def search(self, query: str, top_k: int = 5, pool: int = 20) -> List[RetrievedChunk]:
        sparse_ids = self._sparse_rank(query, pool)
        dense_ids = self._dense_rank(query, pool)

        sparse_rank_of = {idx: r for r, idx in enumerate(sparse_ids)}
        dense_rank_of = {idx: r for r, idx in enumerate(dense_ids)}

        candidate_ids = set(sparse_ids) | set(dense_ids)

        # Reciprocal Rank Fusion
        RRF_K = 60
        fused = []
        for idx in candidate_ids:
            s_rank = sparse_rank_of.get(idx, pool + 1)
            d_rank = dense_rank_of.get(idx, pool + 1)
            score = 1.0 / (RRF_K + s_rank + 1) + 1.0 / (RRF_K + d_rank + 1)
            fused.append((idx, s_rank, d_rank, score))

        fused.sort(key=lambda x: x[3], reverse=True)
        fused_top = fused[: max(top_k * 3, top_k)]

        # rerank the fused shortlist with keyword overlap, then cut to top_k
        reranked = []
        for idx, s_rank, d_rank, fused_score in fused_top:
            chunk = self.chunks[idx]
            rerank_score = self._rerank_score(query, chunk.text)
            reranked.append(
                RetrievedChunk(
                    chunk=chunk,
                    sparse_rank=s_rank,
                    dense_rank=d_rank,
                    fused_score=fused_score,
                    rerank_score=rerank_score,
                )
            )
        reranked.sort(key=lambda r: (r.rerank_score, r.fused_score), reverse=True)
        return reranked[:top_k]
