FROM python:3.11-slim

# System deps: tesseract for OCR fallback, poppler for pdf2image, build tools for xgboost/faiss wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the knowledge base indices at image build time so the container
# starts ready-to-serve rather than building indices on first request.
# Falls back to BM25-only automatically if there's no network access to
# download the embedding model at build time — see scripts/build_knowledge_base.py.
RUN python -m scripts.build_knowledge_base || true

# Train the specialist ML risk models at build time (pure synthetic data,
# no network dependency) so trained models ship inside the image.
RUN python -m src.ml.train

RUN mkdir -p /app/data /app/logs

EXPOSE 8000

ENV ENVIRONMENT=production \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
