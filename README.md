# 🩺 MediScan AI

AI-powered blood report analysis — upload a PDF, get plain-English explanations,
grounded in a real medical knowledge base, with a secondary ML risk-scoring layer.

This project is intentionally over-engineered relative to a typical portfolio
demo, on purpose: it exists to demonstrate production-minded ML engineering
practices (evaluation, structured output, graceful degradation, observability,
security) on top of the core LLM/RAG features, not just "call an LLM API."

## Architecture

```
PDF upload
   │
   ▼
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Extraction               │     │  Rule-based reference-range   │
│  regex + LLM (JSON-mode,  │────▶│  severity scoring              │
│  Pydantic-validated),     │     │  (existing, deterministic)     │
│  confidence-reconciled    │     └──────────────────────────────┘
└─────────────────────────┘                    │
   │                                            ▼
   │                         ┌──────────────────────────────────┐
   │                         │  ML panel router (rule-based)      │
   │                         │  → coverage-gated specialist        │
   │                         │  risk models (metabolic/hematology/ │
   │                         │  liver/renal), SHAP-style attribution│
   │                         └──────────────────────────────────┘
   │                                            │
   ▼                                            ▼
┌────────────────────────────────────────────────────────────────┐
│  Hybrid RAG retrieval: BM25 + dense (FAISS HNSW) → RRF fusion    │
│  → cross-encoder rerank → LLM explanation (cached + traced)      │
└────────────────────────────────────────────────────────────────┘
```

## What's in here

### Core product
- 📤 PDF blood report upload (digital + OCR fallback for scanned reports)
- 🔬 Extracts 35+ lab values via a hybrid regex + LLM pipeline
- 📊 4-tier rule-based severity scoring (normal/borderline/critical/panic)
- 🧠 **4 specialist ML risk models** (metabolic, hematology, liver, renal) — see below
- 🤖 AI explanation in plain English, grounded in a real knowledge base via RAG
- 💬 Follow-up Q&A about your report (streaming)
- 📥 Downloadable PDF summary, report history, JWT auth, admin analytics

### Engineering additions (this iteration)
- ✅ **Populated knowledge base + working RAG** — 8 original medical reference
  docs, chunked and indexed (previously: empty `knowledge_base/`, retrieval
  silently returned nothing)
- ✅ **Hybrid retrieval**: BM25 + dense embeddings fused via Reciprocal Rank
  Fusion, plus a cross-encoder reranker — with automatic fallback to
  BM25-only if the embedding model can't be downloaded (see caveat below)
- ✅ **Golden eval sets + metrics** for both extraction and retrieval
  (`eval/`) — real numbers, not vibes (see **Results** below)
- ✅ **Structured LLM output**: Groq JSON-mode + Pydantic schema validation,
  replacing "prompt for JSON and hope" (`src/schemas.py`)
- ✅ **Confidence scoring**: regex/LLM agreement → high/medium/conflict per
  value, instead of one silently overriding the other
- ✅ **4 specialist ML risk models** (XGBoost) with a rule-based router that
  only runs a model when enough of that panel's tests are present, SHAP-style
  feature attribution, trained on labeled synthetic data (see **ML models**)
- ✅ **Caching**: exact-match + best-effort semantic cache for LLM calls
- ✅ **Lightweight tracing**: structured JSON trace log per LLM call
  (latency, cache hit/miss, success/failure) — `logs/llm_traces.jsonl`
- ✅ **Security hardening**: startup check refuses to boot in production with
  default secrets/wildcard CORS; configurable CORS; rate limiting on
  upload/ask endpoints; portable OCR config (was a hardcoded Windows path)
- ✅ **pytest suite** (41 tests), **Dockerfile**, **docker-compose**, **CI**
  (GitHub Actions: test → build knowledge base → run both evals → train ML
  models → docker build)
- ✅ **Hybrid retrieval**: BM25 + dense embeddings fused via Reciprocal Rank
  Fusion, plus a cross-encoder reranker — verified working end-to-end
  (dense FAISS index + cross-encoder both load and serve real queries)

## ML Risk Models — what they do and why

Rather than one model over every possible lab value (which breaks down
because different blood reports contain wildly different subsets of tests),
this uses **4 specialist models mapped to real clinical panels** —
metabolic, hematology, liver, renal — each with its own small, consistently-
co-occurring feature set. A rule-based router checks feature coverage per
panel and only runs a model when enough of its tests are present; otherwise
it explicitly falls back to the existing rule-based severity scoring.

- **Features**: every lab value is converted to a normalized distance from
  its reference-range midpoint (reuses `src/reference_ranges.py`), so
  different units/scales become comparable.
- **Missing data**: XGBoost handles `NaN` natively (learns a default split
  direction per feature), which is the actual reason gradient boosting was
  chosen over logistic regression here — no manual imputation needed for a
  feature set that's different on every report.
- **Labels**: no public multi-panel lab dataset is reachable from this
  environment, so training data is **synthetic, labeled via established
  multi-criteria clinical rules** (e.g. ATP III-style: 2+ of {high glucose,
  high triglycerides, low HDL} = elevated metabolic risk), with Gaussian
  noise on values and explicit label noise — stated openly rather than
  hidden. Swap in a real cohort dataset (e.g. NHANES) without touching
  `train.py` — the contract is just "dataframe of features + `label`."
- **Explainability**: each prediction returns its top contributing features
  and direction (via XGBoost's additive SHAP-consistent contributions).
- **Framing**: this is a secondary risk-flagging signal surfaced alongside
  the existing rule-based severity, never a replacement or a diagnosis.

## Results (real numbers from this repo — see `eval/`)

| Metric | Result |
|---|---|
| RAG retrieval hit-rate@3 (15 golden queries, hybrid BM25+dense+rerank) | **100%** (15/15) |
| Lab-value extraction — regex-only | **F1 0.957** (P 1.00, R 0.92) |
| Lab-value extraction — hybrid regex+LLM | **F1 0.854** (P 0.76, R 0.97) |
| Metabolic risk model | AUC **0.93**, F1 0.87 |
| Renal risk model | AUC **0.85**, F1 0.62 |
| Hematology risk model | AUC 0.76, F1 0.47 |
| Liver risk model | AUC 0.70, F1 0.41 |

**A genuinely interesting result, reported honestly rather than smoothed
over**: adding the LLM extractor *raised* recall (0.92 → 0.97 — it catches
more values regex misses) but *lowered* precision (1.00 → 0.76 — it also
introduces extra values not in the ground truth, likely from slightly
different test-name normalization or values regex correctly chose to
omit), for a net *lower* F1 than regex-only. This is exactly the kind of
precision/recall tradeoff a merge strategy needs to be tuned for rather
than assumed: naively unioning regex + LLM values is not automatically
better than either alone. The next iteration should either (a) only accept
LLM-only values above a confidence/agreement threshold rather than merging
everything, or (b) use `extract_with_confidence` (already built, see
`src/schemas.py`) to surface conflicts instead of silently including both
sides' values in the final set.

Hematology/liver recall is intentionally left low rather than tuned up
artificially — the underlying synthetic label rate is low (~7-9%) and the
point of publishing this table is to show real, honest eval output, not a
cherry-picked number.

This project was originally built and evaluated in a sandboxed environment
without access to huggingface.co, so the dense embedding index and
cross-encoder reranker couldn't be verified end-to-end there — retrieval
was BM25-only in that setting. **Verified on a normal machine with full
network access**: the dense FAISS index and cross-encoder reranker both
load and serve real queries (`dense_available: True`,
`reranker_available: True`), and the 100% hit-rate above reflects the full
hybrid BM25+dense+rerank pipeline.

## Quick Start

### 1. Setup
```bash
cp .env .your_env
# Fill in GROQ_API_KEY and SECRET_KEY at minimum
pip install -r requirements.txt
```

### 2. Build the knowledge base + train the ML models
```bash
python -m scripts.build_knowledge_base   # chunks + BM25 + dense index
python -m src.ml.train                   # trains the 4 specialist risk models
```

### 3. Install Tesseract OCR (for scanned PDFs)
- **Linux**: `apt-get install tesseract-ocr poppler-utils`
- **macOS**: `brew install tesseract poppler`
- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki, then set
  `TESSERACT_CMD` in `.env` if it's not on PATH.

### 4. Run
```bash
uvicorn src.api:app --reload          # Terminal 1
streamlit run ui.py                   # Terminal 2
```
Or with Docker: `docker compose up --build`

### 5. Run the tests / evals
```bash
python -m pytest tests/ -v
python -m eval.run_rag_eval
python -m eval.run_extraction_eval
```

## Project layout

```
src/
  api.py               FastAPI app — auth, upload, ask, history, analytics
  llm.py               Groq client — retries, JSON-mode, caching, tracing
  llm_extractor.py     LLM-based extraction (JSON-mode + schema validation)
  pdf_extractor.py     Regex-based extraction (+ OCR fallback)
  schemas.py           Pydantic models + confidence reconciliation
  rag_engine.py        Hybrid BM25 + dense retrieval, RRF fusion, reranking
  reference_ranges.py  Deterministic rule-based severity scoring
  disclaimer.py        Panic/critical keyword + threshold detection
  cache.py             Exact-match + semantic LLM response cache
  tracing.py           Structured JSON trace log for every LLM call
  config.py            Central config + production security validation
  ml/
    schemas.py          Panel/feature definitions
    features.py         Normalized-distance feature engineering
    router.py            Coverage-gated panel routing (rule-based)
    data_synthesis.py    Synthetic labeled training data
    train.py              Trains the 4 XGBoost specialist models
    predict.py             Inference + feature attribution
knowledge_base/        Original medical reference docs (chunked for RAG)
eval/                  Golden sets + evaluation harnesses
tests/                 pytest suite (41 tests)
scripts/build_knowledge_base.py   Chunk + index the knowledge base
```

## ⚠️ Disclaimer
MediScan AI is an educational tool only. It does not constitute medical
advice, diagnosis, or treatment — including the ML risk-model output, which
is a secondary pattern-flagging signal, not a diagnosis. Always consult a
qualified healthcare professional before making health decisions.
