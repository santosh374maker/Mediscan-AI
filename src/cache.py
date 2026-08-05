"""
LLM response caching.

Two layers:
  1. Exact-match cache (always available, no ML dependency) — keyed on a
     hash of (system prompt, user prompt, model, temperature). Handles the
     common case of identical repeated questions cheaply.
  2. Semantic cache (best-effort) — if sentence-transformers is available,
     also checks whether a *similar* question was asked before (cosine
     similarity above a threshold) and reuses that answer. Falls back to
     exact-match only if the embedding model can't be loaded (e.g. no
     network access to download it), same graceful-degradation pattern
     used in src/rag_engine.py.

Every cache hit avoids a paid LLM call — this is as much a cost-control
feature as a latency one, and is reported in analytics.
"""
import hashlib
import logging
from typing import Optional

import diskcache

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_cache = diskcache.Cache(str(DATA_DIR / "llm_cache"))
SEMANTIC_SIMILARITY_THRESHOLD = 0.95

_embedding_model = None
_semantic_available = None  # lazy, tri-state: None untested, True/False after first attempt


def _get_embedding_model():
    global _embedding_model, _semantic_available
    if _semantic_available is False:
        return None
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _semantic_available = True
        return _embedding_model
    except Exception as e:
        logger.info("Semantic cache unavailable (%s) — using exact-match cache only.", e)
        _semantic_available = False
        return None


def _hash_key(system: str, prompt: str, model: str, temperature: float) -> str:
    raw = f"{model}|{temperature}|{system}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_response(system: str, prompt: str, model: str, temperature: float) -> Optional[str]:
    exact_key = _hash_key(system, prompt, model, temperature)
    if exact_key in _cache:
        return _cache[exact_key]

    model_obj = _get_embedding_model()
    if model_obj is None:
        return None

    import numpy as np
    semantic_index = _cache.get("__semantic_index__", [])
    if not semantic_index:
        return None

    query_emb = np.array(model_obj.encode([prompt])[0], dtype="float32")
    query_emb /= (np.linalg.norm(query_emb) + 1e-8)

    best_score, best_key = 0.0, None
    for entry in semantic_index:
        emb = np.array(entry["embedding"], dtype="float32")
        score = float(np.dot(query_emb, emb))
        if score > best_score:
            best_score, best_key = score, entry["key"]

    if best_score >= SEMANTIC_SIMILARITY_THRESHOLD and best_key in _cache:
        logger.info("Semantic cache hit (similarity=%.3f)", best_score)
        return _cache[best_key]
    return None


def set_cached_response(system: str, prompt: str, model: str, temperature: float, response: str):
    exact_key = _hash_key(system, prompt, model, temperature)
    _cache[exact_key] = response

    model_obj = _get_embedding_model()
    if model_obj is None:
        return

    import numpy as np
    emb = np.array(model_obj.encode([prompt])[0], dtype="float32")
    emb /= (np.linalg.norm(emb) + 1e-8)

    semantic_index = _cache.get("__semantic_index__", [])
    semantic_index.append({"key": exact_key, "embedding": emb.tolist()})
    # Keep the semantic index bounded — this is a cache, not a database.
    _cache["__semantic_index__"] = semantic_index[-500:]


def cache_stats() -> dict:
    return {
        "entries": len(_cache),
        "semantic_available": _semantic_available is True,
        "volume_path": str(_cache.directory),
    }
