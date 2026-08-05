"""
RAG retrieval evaluation.

Metric: hit-rate@k — for each golden query, was a chunk whose title
contains the expected substring present anywhere in the top-k retrieved
results? This is a lightweight stand-in for full labeled relevance
judgments, appropriate for a knowledge base this size; the same script
scales to a larger, human-labeled relevance set without changing the
harness, just the golden file.

Run:
    python -m eval.run_rag_eval
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag_engine import rag_engine

GOLDEN_PATH = Path(__file__).parent / "golden_rag_queries.json"
RESULTS_PATH = Path(__file__).parent / "results" / "rag_eval_results.json"


def run(top_k: int = 3):
    rag_engine.load()
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    per_query = []
    hits = 0
    for item in golden:
        results = rag_engine.retrieve(item["query"], top_k=top_k)
        titles = [r["title"] for r in results]
        hit = any(item["expected_title_contains"].lower() in t.lower() for t in titles)
        hits += int(hit)
        per_query.append({
            "query": item["query"],
            "expected_title_contains": item["expected_title_contains"],
            "retrieved_titles": titles,
            "hit": hit,
        })

    hit_rate = hits / len(golden) if golden else 0.0
    report = {
        "engine_status": rag_engine.status(),
        "top_k": top_k,
        "n_queries": len(golden),
        "hit_rate_at_k": round(hit_rate, 4),
        "per_query": per_query,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"RAG engine status: {rag_engine.status()}")
    print(f"Hit-rate@{top_k}: {hit_rate:.2%} ({hits}/{len(golden)})")
    for q in per_query:
        mark = "PASS" if q["hit"] else "FAIL"
        print(f"  [{mark}] {q['query']!r} -> {q['retrieved_titles']}")
    print(f"\nSaved full report to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run()
