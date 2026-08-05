from src.pdf_extractor import parse_values_from_text


def test_extracts_simple_colon_separated_value():
    text = "Fasting Glucose: 128 mg/dL"
    values = parse_values_from_text(text)
    assert values.get("glucose fasting") == 128.0


def test_extracts_dash_separated_value():
    text = "HbA1c - 6.9 %"
    values = parse_values_from_text(text)
    assert "hba1c" in values


def test_extracts_multiple_values_from_multiline_report():
    text = (
        "Total Cholesterol : 215 mg/dL\n"
        "LDL Cholesterol - 142 mg/dL\n"
        "HDL Cholesterol: 38 mg/dL\n"
    )
    values = parse_values_from_text(text)
    assert values.get("total cholesterol") == 215.0
    assert values.get("ldl") == 142.0
    assert values.get("hdl") == 38.0


def test_returns_empty_dict_for_text_with_no_values():
    values = parse_values_from_text("This report contains no lab data at all.")
    assert values == {}


def test_handles_empty_string_without_raising():
    values = parse_values_from_text("")
    assert values == {}
