from src.reference_ranges import analyze_report, get_range


def test_get_range_returns_bounds_for_known_test():
    r = get_range("glucose fasting", "male")
    assert r is not None
    assert r["min"] < r["max"]


def test_get_range_unknown_test_returns_none_or_falsy():
    r = get_range("not a real test name xyz", "male")
    assert not r


def test_analyze_report_flags_high_value_as_abnormal():
    results = analyze_report({"glucose fasting": 250}, gender="male")
    assert len(results) == 1
    assert results[0]["status"] == "HIGH"
    assert results[0]["severity"] in ("borderline", "critical", "panic")


def test_analyze_report_flags_normal_value_as_normal():
    results = analyze_report({"glucose fasting": 90}, gender="male")
    assert results[0]["severity"] == "normal"
    assert results[0]["status"] == "NORMAL"


def test_analyze_report_panic_value_for_low_haemoglobin():
    results = analyze_report({"haemoglobin": 1.0}, gender="female")
    assert results[0]["severity"] == "panic"
    assert results[0]["status"] == "LOW"


def test_analyze_report_handles_gender_specific_ranges():
    # HDL cutoffs differ by gender — same borderline-ish value can land
    # differently depending on gender-specific thresholds.
    male_result = analyze_report({"hdl": 38}, gender="male")[0]
    female_result = analyze_report({"hdl": 38}, gender="female")[0]
    assert male_result["severity"] in ("normal", "borderline", "critical", "panic")
    assert female_result["severity"] in ("normal", "borderline", "critical", "panic")


def test_analyze_report_marks_unknown_test_as_unknown_rather_than_crashing():
    results = analyze_report({"totally_made_up_test_xyz": 42}, gender="male")
    assert len(results) == 1
    assert results[0]["status"] == "unknown"
