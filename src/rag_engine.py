"""
RAG Engine — hybrid retrieval over the medical knowledge base.

Architecture:
  1. BM25 (sparse/keyword) retrieval — good at exact terms like "HbA1c" or
     "eGFR" that dense embeddings can sometimes blur together.
  2. Dense retrieval (sentence-transformer embeddings + FAISS HNSW) — good
     at semantic/meaning-based matches, e.g. "sugar levels" -> glucose docs.
  3. Reciprocal Rank Fusion (RRF) combines both rankings into one, which is
     simple, needs no score-scale calibration between BM25 and cosine
     similarity, and is a standard, well-understood fusion technique.
  4. A cross-encoder reranks the fused top-N candidates against the query
     directly (cross-encoders score a (query, doc) pair jointly, which is
     more accurate than comparing independently-computed embeddings, at
     the cost of being too slow to run over the whole corpus — hence
     "retrieve broad with 1-3, then rerank narrow with 4").

Graceful degradation: if the embedding model or FAISS index isn't
available (e.g. no network access to download the model), the engine
automatically falls back to BM25-only retrieval instead of failing. This
matters in real deployments where the vector index build step might run
in a different environment than the serving environment.
"""
import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import EMBEDDING_MODEL, INDEX_PATH, CHUNKS_CSV, TOP_K, DATA_DIR

logger = logging.getLogger(__name__)

BM25_PATH = DATA_DIR / "bm25_index.pkl"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF damping constant


class MedicalRAGEngine:
    def __init__(self):
        self.embedding_model = None
        self.reranker = None
        self.index = None
        self.bm25 = None
        self.documents: List[str] = []
        self.titles: List[str] = []
        self.dense_available = False
        self.reranker_available = False

    def load(self):
        if Path(CHUNKS_CSV).exists():
            df = pd.read_csv(CHUNKS_CSV)
            self.documents = df["chunk"].fillna("").tolist()
            self.titles = df.get("title", pd.Series(["Medical KB"] * len(df))).fillna("Medical KB").tolist()
        else:
            logger.warning("chunks.csv not found — run `python -m scripts.build_knowledge_base` first.")
            self.documents, self.titles = [], []

        if Path(BM25_PATH).exists():
            with open(BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info("BM25 index loaded (%d documents).", len(self.documents))
        elif self.documents:
            from rank_bm25 import BM25Okapi
            tokenized = [d.lower().split() for d in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built in-memory (%d documents).", len(self.documents))

        self._try_load_dense()
        self._try_load_reranker()

    def _try_load_dense(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            if not Path(INDEX_PATH).exists():
                logger.info("No dense FAISS index on disk — running BM25-only until built.")
                return
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.index = faiss.read_index(str(INDEX_PATH))
            self.dense_available = True
            logger.info("Dense FAISS index loaded: %d vectors.", self.index.ntotal)
        except Exception as e:
            logger.warning("Dense retrieval unavailable (%s) — falling back to BM25-only.", e)
            self.dense_available = False

    def _try_load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL)
            self.reranker_available = True
            logger.info("Cross-encoder reranker loaded: %s", RERANKER_MODEL)
        except Exception as e:
            logger.warning("Reranker unavailable (%s) — skipping the rerank stage.", e)
            self.reranker_available = False

    def _bm25_rank(self, query: str, n: int) -> List[int]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        return list(np.argsort(scores)[::-1][:n])

    def _dense_rank(self, query: str, n: int) -> List[int]:
        if not self.dense_available:
            return []
        query_emb = np.array(self.embedding_model.encode([query]), dtype="float32")
        import faiss
        faiss.normalize_L2(query_emb)
        _, indices = self.index.search(query_emb, min(n, len(self.documents)))
        return [int(i) for i in indices[0] if i >= 0]

    def _reciprocal_rank_fusion(self, rankings: List[List[int]]) -> List[int]:
        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[dict]:
        if not self.documents:
            return []

        candidate_pool = max(top_k * 5, 20)
        bm25_ranking = self._bm25_rank(query, candidate_pool)
        dense_ranking = self._dense_rank(query, candidate_pool)

        rankings = [r for r in (bm25_ranking, dense_ranking) if r]
        fused = self._reciprocal_rank_fusion(rankings) if rankings else []
        fused = fused[:candidate_pool]

        if not fused:
            return []

        if self.reranker_available and len(fused) > top_k:
            pairs = [(query, self.documents[i]) for i in fused]
            rerank_scores = self.reranker.predict(pairs)
            order = np.argsort(rerank_scores)[::-1]
            final_indices = [fused[i] for i in order[:top_k]]
            final_scores = [float(rerank_scores[i]) for i in order[:top_k]]
        else:
            final_indices = fused[:top_k]
            final_scores = [1.0 / (rank + 1) for rank in range(len(final_indices))]

        results, seen = [], set()
        for idx, score in zip(final_indices, final_scores):
            chunk = self.documents[idx]
            fingerprint = chunk[:100].strip().lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            results.append({
                "chunk": chunk,
                "title": self.titles[idx] if idx < len(self.titles) else "Medical KB",
                "score": round(score, 4),
            })
        return results

    def build_context(self, results: List[dict]) -> str:
        return "\n\n".join(
            f"[Source: {r['title']} | Relevance: {r['score']:.2f}]\n{r['chunk']}"
            for r in results
        )

    def status(self) -> dict:
        return {
            "documents_loaded": len(self.documents),
            "bm25_available": self.bm25 is not None,
            "dense_available": self.dense_available,
            "reranker_available": self.reranker_available,
        }


rag_engine = MedicalRAGEngine()
