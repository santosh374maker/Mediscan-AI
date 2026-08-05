"""
LLM-based structured extraction — uses Groq's JSON-mode output, validated
against a Pydantic schema, to extract lab values from raw PDF text.

Why this replaces the old "prompt for JSON + json.loads()" approach:
  - `response_format={"type": "json_object"}` constrains the model's output
    at the API level, rather than hoping the model obeys a plain-English
    instruction.
  - The result is still validated against `LLMExtractionResult` — the API
    guarantees *valid JSON*, not the *shape* we asked for, so schema
    validation is still required, not just parsing.
  - Regex and LLM extractions are reconciled into per-value confidence
    records (src/schemas.py) instead of one silently overriding the other.
"""
import json
import logging

from pydantic import ValidationError

from src.llm import call_llm
from src.schemas import LLMExtractionResult, ValueConfidence, reconcile_with_confidence

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a medical data extraction assistant.
Extract every lab test name, numeric value, and unit (if present) from the
provided blood report text.
Respond with ONLY a JSON object of this exact shape, no other keys, no markdown:
{"values": [{"test_name": "haemoglobin", "value": 10.2, "unit": "g/dL"}, ...]}
Use lowercase test names. "value" must be a plain number (no units inside it).
If a test has no discernible numeric value, omit it rather than guessing."""


def extract_values_via_llm(raw_text: str) -> dict:
    """
    Use the LLM (JSON-mode + schema validation) to extract lab values.
    Returns {test_name: float_value}. Falls back to empty dict on any
    failure — a failed LLM extraction should never take down the overall
    pipeline, since regex extraction still runs independently.
    """
    truncated = raw_text[:3000]
    prompt = f"Extract all lab test names and numeric values from this blood report text:\n\n{truncated}"

    try:
        response = call_llm(
            prompt,
            system=EXTRACTION_SYSTEM,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response)
        validated = LLMExtractionResult.model_validate(raw)
        result = validated.as_dict()
        logger.info("LLM extracted %d values (schema-validated).", len(result))
        return result

    except json.JSONDecodeError as e:
        logger.warning("LLM extraction returned invalid JSON despite JSON-mode: %s", e)
        return {}
    except ValidationError as e:
        logger.warning("LLM extraction JSON did not match expected schema: %s", e)
        return {}
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        return {}


def extract_with_fallback(raw_text: str, regex_values: dict) -> dict:
    """
    Merge regex-extracted values with LLM-extracted values.
    Kept for backward compatibility with existing callers that just want a
    flat dict — prefer `extract_with_confidence` for new code, since it
    preserves the agreement/conflict signal instead of discarding it.
    """
    if not raw_text.strip():
        return regex_values
    llm_values = extract_values_via_llm(raw_text)
    merged = {**llm_values, **regex_values}
    logger.info("Merged extraction: %d total values (%d regex, %d LLM).",
                len(merged), len(regex_values), len(llm_values))
    return merged


def extract_with_confidence(raw_text: str, regex_values: dict) -> list[ValueConfidence]:
    """
    Preferred entry point: returns per-value confidence records showing
    whether regex and LLM agreed, and surfacing any conflicts rather than
    silently resolving them.
    """
    llm_values = extract_values_via_llm(raw_text) if raw_text.strip() else {}
    return reconcile_with_confidence(regex_values, llm_values)
