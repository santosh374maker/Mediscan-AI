"""
Panel router — decides which specialist risk model(s) apply to a given
blood report, based on feature coverage.

Deliberately rule-based rather than a learned classifier: routing by
"how many of this panel's tests are present" is fully deterministic,
needs no training data, and is trivially auditable when something looks
wrong. A learned router would add a failure mode without a clear benefit
here — a good example of choosing NOT to reach for ML.

A single report commonly contains multiple panels at once (e.g. a
comprehensive metabolic panel bundles kidney + sugar tests), so the
router returns zero, one, or multiple applicable panels, never a forced
single choice.
"""
from typing import Dict, List

from src.ml.schemas import PANELS
from src.ml.features import coverage


def route(extracted_values: Dict[str, float]) -> List[str]:
    """Return the list of panel names whose coverage gate is satisfied."""
    applicable = []
    for name, spec in PANELS.items():
        if coverage(extracted_values, spec.features) >= spec.min_features_required:
            applicable.append(name)
    return applicable


def route_with_coverage(extracted_values: Dict[str, float]) -> Dict[str, dict]:
    """Same as route(), but also reports coverage detail per panel — used by the API
    to explain to the user why a panel was or wasn't run."""
    report = {}
    for name, spec in PANELS.items():
        n = coverage(extracted_values, spec.features)
        report[name] = {
            "covered_features": n,
            "total_features": len(spec.features),
            "required": spec.min_features_required,
            "eligible": n >= spec.min_features_required,
        }
    return report
