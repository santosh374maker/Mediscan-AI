"""
Disclaimer engine — detects serious/panic conditions in analysis results
and generates appropriate warnings.
"""
from typing import List
from src.config import MEDICAL_DISCLAIMER

PANIC_KEYWORDS = [
    "cancer", "malignant", "malignancy", "tumour", "tumor",
    "stroke", "organ failure", "kidney failure", "liver failure",
    "heart failure", "cardiac arrest", "sepsis", "anemia severe",
    "critical", "emergency", "urgent", "immediate",
]

EMERGENCY_TESTS = {
    "glucose fasting": {"low": 50, "high": 400},
    "blood sugar fasting": {"low": 50, "high": 400},
    "haemoglobin": {"low": 7.0},
    "hemoglobin": {"low": 7.0},
    "platelets": {"low": 50},
    "creatinine": {"high": 10.0},
}


def check_panic_values(analysis_results: List[dict]) -> List[str]:
    """Return list of panic-level test names from analysis results."""
    return [
        r["test_name"]
        for r in analysis_results
        if r.get("severity") == "panic"
    ]


def check_critical_values(analysis_results: List[dict]) -> List[str]:
    """Return list of critical-level test names."""
    return [
        r["test_name"]
        for r in analysis_results
        if r.get("severity") == "critical"
    ]


def check_text_for_keywords(text: str) -> List[str]:
    """Scan AI explanation text for serious medical keywords."""
    text_lower = text.lower()
    return [kw for kw in PANIC_KEYWORDS if kw in text_lower]


def generate_disclaimer(analysis_results: List[dict], ai_text: str = "") -> dict:
    """
    Generate appropriate warning level and message based on results.
    Returns: {level: "safe"|"warning"|"critical"|"emergency", message: str, specialists: list}
    """
    panic_tests = check_panic_values(analysis_results)
    critical_tests = check_critical_values(analysis_results)
    keyword_hits = check_text_for_keywords(ai_text)

    # Collect unique specialists for abnormal results
    specialists = list(set(
        r["specialist"]
        for r in analysis_results
        if r.get("severity") in ("panic", "critical", "borderline")
        and r.get("specialist")
    ))

    if panic_tests:
        return {
            "level": "emergency",
            "message": (
                f"🚨 **EMERGENCY — Seek immediate medical attention.**\n"
                f"The following values are at panic levels: {', '.join(panic_tests).upper()}.\n"
                f"Please go to the nearest emergency room or call emergency services immediately.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            ),
            "specialists": specialists,
        }

    if critical_tests or keyword_hits:
        return {
            "level": "critical",
            "message": (
                f"🔴 **IMPORTANT — Please consult a doctor soon.**\n"
                f"Some of your values are significantly outside the normal range "
                f"and require medical evaluation.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            ),
            "specialists": specialists,
        }

    borderline = [r for r in analysis_results if r.get("severity") == "borderline"]
    if borderline:
        return {
            "level": "warning",
            "message": (
                f"🟡 **Some values are borderline — worth monitoring.**\n"
                f"Consider discussing these with your doctor at your next visit.\n\n"
                f"{MEDICAL_DISCLAIMER}"
            ),
            "specialists": specialists,
        }

    return {
        "level": "safe",
        "message": f"🟢 **All checked values are within normal range.**\n\n{MEDICAL_DISCLAIMER}",
        "specialists": [],
    }
