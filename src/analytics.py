"""
Analytics — logs every query with metadata, exposes stats for dashboard.
"""
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict

from src.config import ANALYTICS_DB

logger = logging.getLogger(__name__)
_lock = Lock()


def _load() -> List[dict]:
    if not Path(ANALYTICS_DB).exists():
        return []
    try:
        with open(ANALYTICS_DB, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save(records: List[dict]):
    with open(ANALYTICS_DB, "w") as f:
        json.dump(records, f, indent=2, default=str)


def log_query(question: str, answer: str, sources: List[str], user: str, latency_ms: float):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "question": question,
        "answer_length": len(answer),
        "sources": sources,
        "latency_ms": round(latency_ms, 1),
    }
    with _lock:
        records = _load()
        records.append(record)
        _save(records)


def get_stats() -> Dict:
    records = _load()
    if not records:
        return {"total_queries": 0, "unique_users": 0, "avg_latency_ms": 0, "top_topics": [], "queries_per_day": {}, "recent_queries": []}

    questions = [r["question"] for r in records]
    users = [r["user"] for r in records]
    latencies = [r["latency_ms"] for r in records]

    all_words = " ".join(questions).lower().split()
    stopwords = {"what", "is", "the", "a", "an", "of", "in", "how", "does", "are", "was", "were", "tell", "me", "about", "my", "have", "i"}
    topic_words = [w for w in all_words if len(w) > 4 and w not in stopwords]
    top_topics = Counter(topic_words).most_common(10)

    day_counts: Counter = Counter()
    for r in records:
        day = r["timestamp"][:10]
        day_counts[day] += 1

    return {
        "total_queries": len(records),
        "unique_users": len(set(users)),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "top_topics": top_topics,
        "queries_per_day": dict(sorted(day_counts.items())[-14:]),
        "recent_queries": records[-10:][::-1],
    }
