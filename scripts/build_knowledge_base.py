"""
Build the RAG knowledge base: chunk every markdown file in knowledge_base/,
then build both retrieval indices used by the hybrid retriever:

  1. BM25 (keyword/sparse) index — pure Python, no network or GPU needed.
  2. Dense FAISS index (sentence-transformers embeddings) — needs network
     access the first time to download the embedding model from
     Hugging Face. If that download isn't available in the current
     environment (e.g. a sandboxed CI runner with restricted egress),
     this script still produces a working BM25-only knowledge base and
     prints a clear warning — the app degrades gracefully to BM25-only
     retrieval rather than failing outright (see src/rag_engine.py).

Run:
    python -m scripts.build_knowledge_base
"""
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (KNOWLEDGE_BASE_DIR, CHUNKS_CSV, CHUNK_SIZE_WORDS,
                         CHUNK_OVERLAP_WORDS, INDEX_PATH, EMBEDDING_MODEL, DATA_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BM25_PATH = DATA_DIR / "bm25_index.pkl"


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE_WORDS,
                overlap: int = CHUNK_OVERLAP_WORDS) -> list:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append({"title": title, "chunk": " ".join(chunk_words)})
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks() -> pd.DataFrame:
    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s — knowledge base will be empty.", kb_dir)
        return pd.DataFrame(columns=["title", "chunk"])

    all_chunks = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        # Chunk per markdown section (## headers) first, so each chunk stays
        # topically coherent, then further split any long section by word count.
        sections = text.split("\n## ")
        for section in sections:
            section_title = section.splitlines()[0].lstrip("# ").strip() if section.strip() else title
            all_chunks.extend(chunk_text(section, f"{title} — {section_title}"))

    df = pd.DataFrame(all_chunks)
    logger.info("Built %d chunks from %d source documents.", len(df), len(md_files))
    return df


def build_bm25(chunks: list) -> BM25Okapi:
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def build_dense_index(chunks: list):
    """Returns a FAISS index, or None if the embedding model can't be loaded
    (e.g. no network access to download it)."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s (requires network on first run)...", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = np.array(model.encode(chunks, show_progress_bar=False), dtype="float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)  # HNSW: sub-linear search, scales far better than flat search
        index.hnsw.efConstruction = 80
        index.add(embeddings)
        return index
    except Exception as e:
        logger.warning(
            "Dense embedding index could not be built (%s). "
            "Continuing with BM25-only retrieval — the app will still work, "
            "just without semantic (meaning-based) search until this is re-run "
            "somewhere with network access to Hugging Face.", e
        )
        return None


def main():
    df = build_chunks()
    df.to_csv(CHUNKS_CSV, index=False)
    logger.info("Wrote chunks to %s", CHUNKS_CSV)

    if df.empty:
        logger.warning("No chunks to index. Add markdown files to knowledge_base/ and re-run.")
        return

    chunks = df["chunk"].tolist()

    bm25 = build_bm25(chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("Wrote BM25 index to %s", BM25_PATH)

    dense_index = build_dense_index(chunks)
    if dense_index is not None:
        import faiss
        faiss.write_index(dense_index, str(INDEX_PATH))
        logger.info("Wrote dense FAISS (HNSW) index to %s", INDEX_PATH)
    else:
        logger.info("Skipped dense index this run — BM25-only mode active.")


if __name__ == "__main__":
    main()
