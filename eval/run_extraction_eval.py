"""
Extraction evaluation.

Scores lab-value extraction against a hand-labeled golden set
(eval/golden_extraction_set.json), reporting precision / recall / F1 for:
  - regex-only extraction (src.pdf_extractor.parse_values_from_text)
  - hybrid regex+LLM extraction (src.llm_extractor.extract_with_fallback),
    only if GROQ_API_KEY is configured — otherwise skipped and clearly
    marked as skipped rather than silently omitted.

Per-value correctness requires both the canonical test name to match AND
the numeric value to match within a small tolerance (handles rounding).

Run:
    python -m eval.run_extraction_eval
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pdf_extractor import parse_values_from_text
from src.config import GROQ_API_KEY

GOLDEN_PATH = Path(__file__).parent / "golden_extraction_set.json"
RESULTS_PATH = Path(__file__).parent / "results" / "extraction_eval_results.json"
TOLERANCE = 0.01


def score_extraction(predicted: dict, ground_truth: dict) -> dict:
    tp = fp = fn = 0
    for key, val in ground_truth.items():
        if key in predicted and abs(predicted[key] - val) <= max(TOLERANCE, abs(val) * 0.01):
            tp += 1
        else:
            fn += 1
    for key in predicted:
        if key not in ground_truth:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate(per_report_scores: list) -> dict:
    tp = sum(s["tp"] for s in per_report_scores)
    fp = sum(s["fp"] for s in per_report_scores)
    fn = sum(s["fn"] for s in per_report_scores)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def run():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    regex_scores = []
    hybrid_scores = []
    hybrid_available = bool(GROQ_API_KEY)

    if not hybrid_available:
        print("GROQ_API_KEY not set — skipping hybrid (regex+LLM) evaluation, "
              "scoring regex-only extraction. Set GROQ_API_KEY and re-run for the full comparison.")

    per_report_detail = []
    for item in golden:
        regex_pred = parse_values_from_text(item["text"])
        r_score = score_extraction(regex_pred, item["ground_truth"])
        regex_scores.append(r_score)

        detail = {"id": item["id"], "regex": r_score, "regex_predicted": regex_pred}

        if hybrid_available:
            from src.llm_extractor import extract_with_fallback
            hybrid_pred = extract_with_fallback(item["text"], regex_pred)
            h_score = score_extraction(hybrid_pred, item["ground_truth"])
            hybrid_scores.append(h_score)
            detail["hybrid"] = h_score
            detail["hybrid_predicted"] = hybrid_pred

        per_report_detail.append(detail)

    report = {
        "n_reports": len(golden),
        "regex_only": aggregate(regex_scores),
        "hybrid_regex_llm": aggregate(hybrid_scores) if hybrid_available else "skipped (no GROQ_API_KEY)",
        "per_report": per_report_detail,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nRegex-only:   P={report['regex_only']['precision']:.3f}  "
          f"R={report['regex_only']['recall']:.3f}  F1={report['regex_only']['f1']:.3f}")
    if hybrid_available:
        print(f"Hybrid (R+L): P={report['hybrid_regex_llm']['precision']:.3f}  "
              f"R={report['hybrid_regex_llm']['recall']:.3f}  F1={report['hybrid_regex_llm']['f1']:.3f}")
    print(f"\nSaved full report to {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    run()
