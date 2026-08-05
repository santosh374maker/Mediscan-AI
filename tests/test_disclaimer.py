from src.disclaimer import generate_disclaimer


def _result(test_name, severity, specialist="Doctor"):
    return {"test_name": test_name, "severity": severity, "specialist": specialist,
            "status": "HIGH", "value": 1, "unit": ""}


def test_panic_value_produces_emergency_level():
    results = [_result("haemoglobin", "panic")]
    d = generate_disclaimer(results)
    assert d["level"] == "emergency"


def test_critical_value_produces_critical_level():
    results = [_result("glucose fasting", "critical")]
    d = generate_disclaimer(results)
    assert d["level"] == "critical"


def test_borderline_only_produces_warning_level():
    results = [_result("total cholesterol", "borderline")]
    d = generate_disclaimer(results)
    assert d["level"] == "warning"


def test_all_normal_produces_safe_level():
    results = [{"test_name": "glucose fasting", "severity": "normal", "specialist": None,
                "status": "NORMAL", "value": 90, "unit": "mg/dL"}]
    d = generate_disclaimer(results)
    assert d["level"] == "safe"


def test_panic_takes_priority_over_critical_and_borderline():
    results = [_result("a", "borderline"), _result("b", "critical"), _result("c", "panic")]
    d = generate_disclaimer(results)
    assert d["level"] == "emergency"


def test_keyword_in_ai_text_escalates_to_critical():
    results = [{"test_name": "glucose fasting", "severity": "normal", "specialist": None,
                "status": "NORMAL", "value": 90, "unit": "mg/dL"}]
    d = generate_disclaimer(results, ai_text="this could indicate a possible malignancy")
    assert d["level"] == "critical"


def test_disclaimer_always_includes_medical_disclaimer_text():
    results = []
    d = generate_disclaimer(results)
    assert "educational" in d["message"].lower()
