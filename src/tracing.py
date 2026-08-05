"""
Lightweight LLM observability.

Not a replacement for a full platform like Langfuse/LangSmith — those need
an external account/API key this project shouldn't assume every reviewer
has. Instead, this writes structured JSON trace records to
logs/llm_traces.jsonl, one line per LLM call, with:
  - a request id + span timing (start, end, duration_ms)
  - the prompt/system prompt lengths (proxy for token cost, no need for a
    tokenizer dependency just for a rough estimate)
  - cache hit/miss
  - success/failure and error type if it failed

This is intentionally swappable: point at a real tracing backend later
(Langfuse, LangSmith, an OpenTelemetry exporter) without touching call
sites — everything funnels through `record_trace()`.
"""
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR

logger = logging.getLogger(__name__)

TRACE_LOG_PATH = Path(LOGS_DIR) / "llm_traces.jsonl"
_lock = threading.Lock()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def record_trace(record: dict):
    record.setdefault("timestamp", time.time())
    try:
        with _lock:
            with open(TRACE_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to write trace record: %s", e)


@contextmanager
def trace_span(operation: str, request_id: Optional[str] = None, **metadata):
    """
    Usage:
        with trace_span("llm_call", model=GROQ_MODEL, cache_hit=False) as span:
            result = do_the_thing()
            span["output_length"] = len(result)
    """
    request_id = request_id or new_request_id()
    start = time.time()
    span = {"request_id": request_id, "operation": operation, **metadata}
    error = None
    try:
        yield span
    except Exception as e:
        error = str(e)
        raise
    finally:
        span["duration_ms"] = round((time.time() - start) * 1000, 2)
        span["success"] = error is None
        if error:
            span["error"] = error
        record_trace(span)


def read_recent_traces(n: int = 50) -> list:
    if not TRACE_LOG_PATH.exists():
        return []
    with open(TRACE_LOG_PATH) as f:
        lines = f.readlines()[-n:]
    return [json.loads(line) for line in lines if line.strip()]
