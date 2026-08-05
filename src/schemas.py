"""
Pydantic schemas for structured extraction output.

Replaces "prompt the model to return JSON, then json.loads() and hope" with
a validated schema. Groq's OpenAI-compatible API supports constrained JSON
output (response_format={"type": "json_object"}) — combined with a Pydantic
model to validate the *shape* of what comes back, this eliminates a whole
class of parsing failures (missing keys, wrong types, stray text) rather
than catching them after the fact with regex stripping.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ExtractedLabValue(BaseModel):
    test_name: str = Field(..., description="Lowercase canonical-ish test name, e.g. 'haemoglobin'")
    value: float
    unit: Optional[str] = None

    @field_validator("test_name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().lower()


class LLMExtractionResult(BaseModel):
    """Schema the LLM's JSON output must conform to."""
    values: List[ExtractedLabValue] = Field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        return {item.test_name: item.value for item in self.values}


class ValueConfidence(BaseModel):
    """
    Per-value confidence, derived from regex/LLM agreement rather than a
    single source's say-so:
      - "high":   both regex and LLM extracted the same value (agreement)
      - "medium": only one extraction method found it
      - "conflict": both found it, but the values disagree — surfaced to
        the caller instead of silently picking one, since silently
        overriding on disagreement is how extraction bugs hide.
    """
    test_name: str
    value: float
    confidence: str  # "high" | "medium" | "conflict"
    sources: List[str]  # subset of ["regex", "llm"]
    conflicting_value: Optional[float] = None


def reconcile_with_confidence(regex_values: Dict[str, float],
                               llm_values: Dict[str, float],
                               tolerance_pct: float = 0.02) -> List[ValueConfidence]:
    """
    Merge regex + LLM extractions into per-value confidence records instead
    of silently overriding one with the other. This is the piece the
    original hybrid extractor was missing: agreement is a signal worth
    keeping, and disagreement is worth surfacing, not hiding.
    """
    all_keys = set(regex_values) | set(llm_values)
    results = []
    for key in sorted(all_keys):
        in_regex = key in regex_values
        in_llm = key in llm_values

        if in_regex and in_llm:
            r_val, l_val = regex_values[key], llm_values[key]
            if abs(r_val - l_val) <= max(tolerance_pct * abs(r_val), 1e-6):
                results.append(ValueConfidence(
                    test_name=key, value=r_val, confidence="high",
                    sources=["regex", "llm"],
                ))
            else:
                # Regex wins as the reported value (more precise for
                # well-formatted reports per the original design), but the
                # conflict is surfaced rather than silently dropped.
                results.append(ValueConfidence(
                    test_name=key, value=r_val, confidence="conflict",
                    sources=["regex", "llm"], conflicting_value=l_val,
                ))
        elif in_regex:
            results.append(ValueConfidence(
                test_name=key, value=regex_values[key], confidence="medium", sources=["regex"],
            ))
        else:
            results.append(ValueConfidence(
                test_name=key, value=llm_values[key], confidence="medium", sources=["llm"],
            ))
    return results
