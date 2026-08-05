"""
Centralized configuration — loads from .env, never hardcoded secrets.
"""
import logging
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # "development" | "production"

# ── LLM ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"  # MediScan AI — medical analysis model
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Auth ─────────────────────────────────────────────
_DEFAULT_INSECURE_SECRET = "changeme-insecure-default"
SECRET_KEY = os.getenv("SECRET_KEY", _DEFAULT_INSECURE_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
_DEFAULT_INSECURE_PASSWORD = "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", _DEFAULT_INSECURE_PASSWORD)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

# ── CORS ─────────────────────────────────────────────
# Comma-separated list in .env, e.g. CORS_ORIGINS=https://myapp.com,https://admin.myapp.com
# Defaults to localhost-only for dev; production must set this explicitly (enforced below).
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501")
CORS_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

# ── Rate limiting ────────────────────────────────────
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "10/minute")
RATE_LIMIT_ASK = os.getenv("RATE_LIMIT_ASK", "20/minute")

# ── Database ─────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}",
)

# ── RAG ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
MAX_MEMORY = 10

# ── Paths ─────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
INDEX_PATH = DATA_DIR / "vector_db.index"
CHUNKS_CSV = DATA_DIR / "chunks.csv"
ANALYTICS_DB = DATA_DIR / "analytics.json"
LOGS_DIR = BASE_DIR / "logs"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

for d in [DATA_DIR, UPLOADS_DIR, REPORTS_DIR, LOGS_DIR, KNOWLEDGE_BASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── MediScan Settings ─────────────────────────────────
APP_NAME = "MediScan AI"
APP_DESCRIPTION = "AI-powered blood report analysis"

BORDERLINE_THRESHOLD = 0.15
CRITICAL_THRESHOLD = 0.30

MEDICAL_DISCLAIMER = (
    "⚠️ MediScan AI is an educational tool only. "
    "It does not constitute medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional before "
    "making any health decisions."
)

# ── OCR ──────────────────────────────────────────────
# Configurable via env instead of a hardcoded Windows path — the previous
# hardcoded `C:\Program Files\Tesseract-OCR\tesseract.exe` silently broke
# OCR on every non-Windows machine (including this project's own Docker
# image). On Linux/macOS with tesseract installed via the system package
# manager, pytesseract finds it on PATH automatically and no override is
# needed at all.
_tesseract_cmd = os.getenv("TESSERACT_CMD", "")
try:
    import pytesseract
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
except ImportError:
    pass


# ── Startup security checks ───────────────────────────
def validate_production_security():
    """
    Refuse to boot with insecure defaults in production. Called explicitly
    from src/api.py on startup rather than at import time, so tests and
    local dev tooling that import src.config don't get blocked.
    """
    problems = []
    if ENVIRONMENT == "production":
        if SECRET_KEY == _DEFAULT_INSECURE_SECRET:
            problems.append("SECRET_KEY is still the insecure default — set a real secret in .env")
        if ADMIN_PASSWORD == _DEFAULT_INSECURE_PASSWORD:
            problems.append("ADMIN_PASSWORD is still the insecure default 'admin' — set a real password in .env")
        if "*" in CORS_ORIGINS:
            problems.append("CORS_ORIGINS allows '*' in production — set explicit allowed origins in .env")
        if not GROQ_API_KEY:
            problems.append("GROQ_API_KEY is not set — LLM features will fail")

    if problems:
        message = "Refusing to start in production with insecure configuration:\n  - " + "\n  - ".join(problems)
        raise RuntimeError(message)

    if SECRET_KEY == _DEFAULT_INSECURE_SECRET:
        logger.warning("SECRET_KEY is using the insecure default — fine for local dev, "
                        "must be overridden before deploying (set ENVIRONMENT=production to enforce this).")
