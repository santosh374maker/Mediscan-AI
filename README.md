# 🩺 MediScan AI

**An AI-powered blood report analyzer that explains lab results in plain English — grounded in a real medical knowledge base, backed by trained risk-classification models, and built with the evaluation, security, and observability practices a production ML system actually needs.**

This project is intentionally more rigorously engineered than a typical portfolio demo. The goal isn't just "call an LLM on a PDF" — it's to demonstrate the full discipline around that: measured accuracy, structured/validated output, graceful degradation, and a system that's honest about its own limitations.

---

## 🚀 Live Demo & Links

- 🌐 **Live Application**: Not currently deployed — see [Getting Started](#-getting-started) to run it locally in a few minutes.
- 📝 **API Documentation**: Auto-generated Swagger UI at `http://localhost:8000/docs` once the server is running.
- 💻 **Repository**: [github.com/santosh374maker/mediscan-ai](https://github.com/santosh374maker/mediscan-ai)

---

## ✨ Features

- 🔎 **Hybrid RAG retrieval** — BM25 (keyword) + dense FAISS embeddings fused via Reciprocal Rank Fusion, then reranked with a cross-encoder. Verified at **100% hit-rate@3** on a 15-query golden retrieval set.
- 🧪 **Hybrid extraction pipeline** — regex + LLM (Groq JSON-mode, Pydantic-validated), with agreement-based confidence scoring (`high` / `medium` / `conflict`) instead of one method silently overriding the other.
- 🧠 **4 specialist ML risk models** (metabolic, hematology, liver, renal) — XGBoost classifiers behind a rule-based, coverage-gated router that only runs a model when a report actually has enough of that panel's tests present.
- ⚖️ **Class-imbalance-aware training** — `scale_pos_weight` + F-beta-tuned per-panel decision thresholds, addressing a real, diagnosed recall weakness rather than shipping the naive 0.5 cutoff.
- 📊 **4-tier deterministic severity scoring** (normal / borderline / critical / panic) as the primary, explainable signal — the ML models are a secondary flag, never a replacement.
- 🔐 **Security-hardened by default** — JWT auth, startup checks that refuse to boot in production with default secrets or wildcard CORS, and rate limiting on upload/ask endpoints.
- ⚡ **Caching + observability** — exact-match and best-effort semantic LLM response caching, plus structured JSON trace logs (latency, cache hit/miss, success/failure) for every LLM call.
- ✅ **Golden evaluation sets** for both retrieval and extraction, with real, published numbers — not assumed accuracy.

---

## 🛠️ Tech Stack

| Category | Technologies Used |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn, Pydantic |
| **ML / Data Science** | XGBoost, scikit-learn, NumPy, pandas |
| **RAG / LLM** | Groq API, FAISS (HNSW), `rank-bm25`, `sentence-transformers` (bi-encoder + cross-encoder reranker) |
| **Database** | SQLAlchemy, SQLite |
| **Auth & Security** | JWT (`python-jose`), `passlib`/`bcrypt`, `slowapi` (rate limiting) |
| **Caching** | `diskcache` (exact-match + semantic cache) |
| **Frontend** | HTML5, CSS3, vanilla JavaScript — served directly by FastAPI, same-origin |
| **PDF / OCR** | `pdfplumber`, `pytesseract`, `pdf2image`, `reportlab` |
| **DevOps / Testing** | Docker, Docker Compose, GitHub Actions (CI/CD), pytest (41 tests) |

---

## 🎯 Scope

MediScan AI is built as an **educational, portfolio-grade demonstration** of a production-style ML/AI engineering pipeline applied to a healthcare-adjacent use case — not a clinical or diagnostic product. It's designed to show:

- How to combine deterministic rule-based logic with LLM-generated explanations without letting the LLM "make things up" ungrounded
- How to handle a real-world ML constraint (every input document has a different, inconsistent set of features) with a coverage-gated model router rather than one brittle universal model
- How to build and *measure* a RAG pipeline, rather than assuming retrieval quality
- How to layer in the engineering practices (evaluation, security, observability, testing, CI/CD) that separate a working demo from something closer to production-ready

It is explicitly **not** intended to diagnose, treat, or replace consultation with a healthcare professional.

---

## ⚙️ Architecture & System Design

```
PDF upload
   │
   ▼
┌───────────────────────────┐     ┌──────────────────────────────┐
│  Extraction                 │     │  Rule-based reference-range   │
│  regex + LLM (JSON-mode,    │────▶│  severity scoring              │
│  Pydantic-validated),       │     │  (deterministic, explainable)  │
│  confidence-reconciled      │     └──────────────────────────────┘
└───────────────────────────┘                    │
   │                                              ▼
   │                           ┌──────────────────────────────────┐
   │                           │  ML panel router (rule-based)      │
   │                           │  → coverage-gated specialist        │
   │                           │  risk models (metabolic/hematology/ │
   │                           │  liver/renal) + feature attribution │
   │                           └──────────────────────────────────┘
   │                                              │
   ▼                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Hybrid RAG retrieval: BM25 + dense (FAISS HNSW) → RRF fusion      │
│  → cross-encoder rerank → LLM explanation (cached + traced)        │
└──────────────────────────────────────────────────────────────────┘
```

### ML Risk Models — what they do and why

Rather than one model over every possible lab value (which breaks down because different blood reports contain wildly different subsets of tests), this uses **4 specialist models mapped to real clinical panels** — metabolic, hematology, liver, renal — each with its own small, consistently co-occurring feature set. A rule-based router checks feature coverage per panel and only runs a model when enough of its tests are present; otherwise it explicitly falls back to rule-based severity scoring alone.

- **Features**: every lab value is converted to a normalized distance from its reference-range midpoint, so different units/scales become comparable.
- **Missing data**: XGBoost handles `NaN` natively (learns a default split direction per feature) — the actual reason gradient boosting was chosen over logistic regression here, since a manually-imputed feature set would need to change on every single report.
- **Labels**: training data is **synthetic**, labeled via established multi-criteria clinical rules (e.g. ATP III-style: 2+ of {high glucose, high triglycerides, low HDL} = elevated metabolic risk), with Gaussian and explicit label noise — stated openly, not hidden. Swappable for a real cohort dataset (e.g. NHANES) without touching the training code.
- **Class imbalance handling**: `scale_pos_weight` during training plus an F-beta-tuned decision threshold per panel (rather than a hardcoded 0.5), applied only to the panels with a documented recall weakness.
- **Explainability**: each prediction returns its top contributing features and direction.
- **Framing**: a secondary risk-flagging signal surfaced alongside rule-based severity — never a replacement or a diagnosis.

---

## 📊 Results

*(Real numbers from this repo — reproducible via `eval/` and `src/ml/train.py`.)*

| Metric | Result |
|---|---|
| RAG retrieval hit-rate@3 (15 golden queries, hybrid BM25+dense+rerank) | **100%** (15/15) |
| Lab-value extraction — regex-only | **F1 0.957** (P 1.00, R 0.92) |
| Lab-value extraction — hybrid regex+LLM | F1 0.854 (P 0.76, R 0.97) |
| Metabolic risk model | AUC **0.935**, F1 0.870 |
| Renal risk model | AUC 0.839, F1 0.648 (recall 0.473 → **0.630** after imbalance fix) |
| Hematology risk model | AUC 0.742, F1 0.452 (recall 0.330 → **0.487** after imbalance fix) |
| Liver risk model | AUC 0.707, F1 0.343 (recall 0.265 → **0.469** after imbalance fix) |

**Two findings reported honestly rather than smoothed over:**

1. The hybrid regex+LLM extractor *raises recall* (0.92 → 0.97) but *lowers precision* (1.00 → 0.76), for a net lower F1 than regex alone — naively unioning two extraction methods is not automatically better than either one; it needs a tuned merge strategy (`extract_with_confidence` in `src/schemas.py` exists for exactly this, and is the next thing to wire in).
2. Hematology and liver's AUC (0.70–0.74) puts a real ceiling on how much recall can be recovered through threshold tuning alone — the imbalance fix roughly doubled recall on both, but further improvement likely needs better features or more/real training data, not just a different cutoff.

---

## ✅ Advantages

- **Graceful degradation everywhere** — if the dense embedding model can't be loaded, retrieval falls back to BM25 automatically; if a report doesn't have enough tests for a risk model, it says so explicitly rather than guessing.
- **Every claim in this README is backed by a script you can run yourself** (`eval/run_rag_eval.py`, `eval/run_extraction_eval.py`, `src/ml/train.py`) — no unverifiable accuracy claims.
- **Confidence, not silent overriding** — when regex and LLM extraction disagree, that conflict is surfaced, not hidden behind whichever method ran last.
- **Security is opt-out-proof, not opt-in** — the app refuses to start in production with default secrets or a wildcard CORS policy, rather than relying on a developer remembering to change them.
- **Two real bugs were found and fixed during development** (a train/inference feature-space mismatch that would have made every ML prediction meaningless, and a regex bug that mis-parsed test names containing digits) — both documented rather than quietly patched.

## ⚠️ Limitations

- **Training data for the ML risk models is synthetic**, not real patient records — a deliberate, disclosed simplification, not a hidden one. Numbers demonstrate the *technique*, not real-world clinical accuracy.
- **PDF extraction accuracy varies by report format** — the regex extractor is tuned against a specific golden set; different lab formats will need broader eval coverage.
- **Hybrid regex+LLM extraction currently has lower F1 than regex alone** (see Results) — a known, unresolved tradeoff, not yet fixed by the confidence-reconciliation logic that already exists in the codebase.
- **This is not a diagnostic tool**, and the ML risk-model output is explicitly a secondary pattern-flagging signal, never a replacement for professional medical judgment.
- **No large-scale real-user testing yet** — evaluated against self-authored golden sets, not a broad or adversarial real-world sample.

---

## 🏁 Getting Started

### Prerequisites
```bash
python3 --version   # Requires Python 3.11+
pip --version
```

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/santosh374maker/mediscan-ai
cd mediscan-ai
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables:**
```bash
cp .env.example .env
```
Then fill in, at minimum, `GROQ_API_KEY` and `SECRET_KEY` in `.env`.

**4. Build the knowledge base and train the ML models:**
```bash
python -m scripts.build_knowledge_base   # chunks + BM25 + dense index
python -m src.ml.train                   # trains the 4 specialist risk models
```

**5. (Optional) Install Tesseract OCR — needed only for scanned PDFs:**
- **Linux**: `apt-get install tesseract-ocr poppler-utils`
- **macOS**: `brew install tesseract poppler`
- **Windows**: [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env` if it's not on PATH.

**6. Run the app:**
```bash
uvicorn src.api:app --reload
```
Then open **http://localhost:8000/app/** — the dashboard is served directly by FastAPI, no separate frontend server needed.

Or with Docker:
```bash
docker compose up --build
```

---

## 🧪 Running Tests

```bash
# Full pytest suite (41 tests)
python -m pytest tests/ -v

# RAG retrieval evaluation
python -m eval.run_rag_eval

# Lab-value extraction evaluation
python -m eval.run_extraction_eval
```

---

## 📁 Project Layout

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
static/index.html      HTML/CSS/JS dashboard frontend
knowledge_base/         Original medical reference docs (chunked for RAG)
eval/                   Golden sets + evaluation harnesses
tests/                  pytest suite (41 tests)
scripts/build_knowledge_base.py   Chunk + index the knowledge base
```

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. *(Add a `LICENSE` file to the repo root if one isn't already present.)*

---

## ✉️ Contact

**Santosh Achary** — [acharysantosh19@gmail.com](mailto:acharysantosh19@gmail.com)
GitHub: [@santosh374maker](https://github.com/santosh374maker) · LinkedIn: [s-santosh-achary](https://linkedin.com/in/s-santosh-achary)

---

## ⚠️ Medical Disclaimer

MediScan AI is an educational tool only. It does not constitute medical advice, diagnosis, or treatment — including the ML risk-model output, which is a secondary pattern-flagging signal, not a diagnosis. Always consult a qualified healthcare professional before making health decisions.