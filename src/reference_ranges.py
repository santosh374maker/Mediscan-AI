"""
Reference ranges for common blood tests.
Covers CBC, liver function, kidney function, blood sugar, thyroid, lipids.
"""
from src.config import BORDERLINE_THRESHOLD, CRITICAL_THRESHOLD

REFERENCE_RANGES = {
    "haemoglobin": {
        "male":   {"min": 13.5, "max": 17.5},
        "female": {"min": 12.0, "max": 15.5},
        "unit": "g/dL", "category": "CBC",
        "meaning": "Carries oxygen in your blood. Low levels may indicate anaemia.",
        "specialist": "Haematologist",
    },
    "hemoglobin": {
        "male":   {"min": 13.5, "max": 17.5},
        "female": {"min": 12.0, "max": 15.5},
        "unit": "g/dL", "category": "CBC",
        "meaning": "Carries oxygen in your blood. Low levels may indicate anaemia.",
        "specialist": "Haematologist",
    },
    "wbc": {
        "min": 4.0, "max": 11.0, "unit": "10³/µL", "category": "CBC",
        "meaning": "White blood cells fight infection. High may indicate infection or inflammation.",
        "specialist": "Haematologist",
    },
    "rbc": {
        "male":   {"min": 4.5, "max": 5.9},
        "female": {"min": 4.0, "max": 5.2},
        "unit": "10⁶/µL", "category": "CBC",
        "meaning": "Red blood cells carry oxygen. Low may indicate anaemia.",
        "specialist": "Haematologist",
    },
    "platelets": {
        "min": 150, "max": 400, "unit": "10³/µL", "category": "CBC",
        "meaning": "Help blood clot. Low increases bleeding risk, high increases clot risk.",
        "specialist": "Haematologist",
    },
    "hematocrit": {
        "male":   {"min": 41, "max": 53},
        "female": {"min": 36, "max": 46},
        "unit": "%", "category": "CBC",
        "meaning": "Percentage of blood made up of red blood cells.",
        "specialist": "Haematologist",
    },
    "mcv": {
        "min": 80, "max": 100, "unit": "fL", "category": "CBC",
        "meaning": "Size of red blood cells. Abnormal size hints at type of anaemia.",
        "specialist": "Haematologist",
    },
    "mch": {
        "min": 27, "max": 33, "unit": "pg", "category": "CBC",
        "meaning": "Amount of haemoglobin per red blood cell.",
        "specialist": "Haematologist",
    },
    "neutrophils": {
        "min": 40, "max": 70, "unit": "%", "category": "CBC",
        "meaning": "First responders to bacterial infection.",
        "specialist": "Haematologist",
    },
    "lymphocytes": {
        "min": 20, "max": 40, "unit": "%", "category": "CBC",
        "meaning": "Fight viral infections and produce antibodies.",
        "specialist": "Haematologist",
    },
    "glucose fasting": {
        "min": 70, "max": 100, "unit": "mg/dL", "category": "Blood Sugar",
        "meaning": "Blood sugar after fasting. High levels may indicate diabetes.",
        "specialist": "Endocrinologist",
    },
    "blood sugar fasting": {
        "min": 70, "max": 100, "unit": "mg/dL", "category": "Blood Sugar",
        "meaning": "Blood sugar after fasting. High levels may indicate diabetes.",
        "specialist": "Endocrinologist",
    },
    "hba1c": {
        "min": 4.0, "max": 5.6, "unit": "%", "category": "Blood Sugar",
        "meaning": "Average blood sugar over 3 months. Key diabetes marker.",
        "specialist": "Endocrinologist",
    },
    "postprandial glucose": {
        "min": 70, "max": 140, "unit": "mg/dL", "category": "Blood Sugar",
        "meaning": "Blood sugar 2 hours after eating.",
        "specialist": "Endocrinologist",
    },
    "creatinine": {
        "male":   {"min": 0.7, "max": 1.2},
        "female": {"min": 0.5, "max": 1.0},
        "unit": "mg/dL", "category": "Kidney Function",
        "meaning": "Waste product filtered by kidneys. High levels may indicate kidney disease.",
        "specialist": "Nephrologist",
    },
    "urea": {
        "min": 7, "max": 20, "unit": "mg/dL", "category": "Kidney Function",
        "meaning": "Waste product filtered by kidneys.",
        "specialist": "Nephrologist",
    },
    "bun": {
        "min": 7, "max": 20, "unit": "mg/dL", "category": "Kidney Function",
        "meaning": "Blood urea nitrogen — measures kidney filtering ability.",
        "specialist": "Nephrologist",
    },
    "uric acid": {
        "male":   {"min": 3.4, "max": 7.0},
        "female": {"min": 2.4, "max": 6.0},
        "unit": "mg/dL", "category": "Kidney Function",
        "meaning": "High levels can cause gout and kidney stones.",
        "specialist": "Nephrologist",
    },
    "egfr": {
        "min": 60, "max": 120, "unit": "mL/min/1.73m²", "category": "Kidney Function",
        "meaning": "Estimated kidney filtration rate. Below 60 may indicate kidney disease.",
        "specialist": "Nephrologist",
    },
    "alt": {
        "min": 7, "max": 56, "unit": "U/L", "category": "Liver Function",
        "meaning": "Liver enzyme. High levels may indicate liver damage or disease.",
        "specialist": "Gastroenterologist",
    },
    "ast": {
        "min": 10, "max": 40, "unit": "U/L", "category": "Liver Function",
        "meaning": "Liver/heart enzyme. High may indicate liver or heart damage.",
        "specialist": "Gastroenterologist",
    },
    "alkaline phosphatase": {
        "min": 44, "max": 147, "unit": "U/L", "category": "Liver Function",
        "meaning": "Enzyme found in liver and bones. High may indicate liver or bone disease.",
        "specialist": "Gastroenterologist",
    },
    "bilirubin total": {
        "min": 0.1, "max": 1.2, "unit": "mg/dL", "category": "Liver Function",
        "meaning": "Breakdown product of red blood cells. High causes jaundice.",
        "specialist": "Gastroenterologist",
    },
    "albumin": {
        "min": 3.4, "max": 5.4, "unit": "g/dL", "category": "Liver Function",
        "meaning": "Protein made by liver. Low may indicate liver disease or malnutrition.",
        "specialist": "Gastroenterologist",
    },
    "tsh": {
        "min": 0.4, "max": 4.0, "unit": "mIU/L", "category": "Thyroid",
        "meaning": "Controls thyroid function. High may indicate hypothyroidism.",
        "specialist": "Endocrinologist",
    },
    "t3": {
        "min": 80, "max": 200, "unit": "ng/dL", "category": "Thyroid",
        "meaning": "Active thyroid hormone. Controls metabolism.",
        "specialist": "Endocrinologist",
    },
    "t4": {
        "min": 5.0, "max": 12.0, "unit": "µg/dL", "category": "Thyroid",
        "meaning": "Thyroid hormone converted to T3 in the body.",
        "specialist": "Endocrinologist",
    },
    "free t4": {
        "min": 0.8, "max": 1.8, "unit": "ng/dL", "category": "Thyroid",
        "meaning": "Unbound thyroid hormone available for use by the body.",
        "specialist": "Endocrinologist",
    },
    "total cholesterol": {
        "min": 0, "max": 200, "unit": "mg/dL", "category": "Lipids",
        "meaning": "Total fat in blood. High increases heart disease risk.",
        "specialist": "Cardiologist",
    },
    "ldl": {
        "min": 0, "max": 100, "unit": "mg/dL", "category": "Lipids",
        "meaning": "Bad cholesterol. High increases plaque buildup in arteries.",
        "specialist": "Cardiologist",
    },
    "hdl": {
        "male":   {"min": 40, "max": 999},
        "female": {"min": 50, "max": 999},
        "unit": "mg/dL", "category": "Lipids",
        "meaning": "Good cholesterol. Higher is better.",
        "specialist": "Cardiologist",
    },
    "triglycerides": {
        "min": 0, "max": 150, "unit": "mg/dL", "category": "Lipids",
        "meaning": "Type of fat in blood. High increases heart disease and pancreatitis risk.",
        "specialist": "Cardiologist",
    },
    "serum iron": {
        "male":   {"min": 65, "max": 175},
        "female": {"min": 50, "max": 170},
        "unit": "µg/dL", "category": "Iron Studies",
        "meaning": "Iron level in blood. Low may indicate iron deficiency anaemia.",
        "specialist": "Haematologist",
    },
    "ferritin": {
        "male":   {"min": 12, "max": 300},
        "female": {"min": 12, "max": 150},
        "unit": "ng/mL", "category": "Iron Studies",
        "meaning": "Iron storage protein. Low indicates iron deficiency.",
        "specialist": "Haematologist",
    },
    "tibc": {
        "min": 250, "max": 370, "unit": "µg/dL", "category": "Iron Studies",
        "meaning": "Total iron binding capacity. High with low iron suggests deficiency.",
        "specialist": "Haematologist",
    },
}

PANIC_VALUES = {
    "glucose fasting":     {"low": 50,  "high": 400},
    "blood sugar fasting": {"low": 50,  "high": 400},
    "haemoglobin":         {"low": 7.0, "high": 20.0},
    "hemoglobin":          {"low": 7.0, "high": 20.0},
    "platelets":           {"low": 50,  "high": 1000},
    "creatinine":          {"high": 10.0},
    "bilirubin total":     {"high": 15.0},
    "alt":                 {"high": 1000},
    "ast":                 {"high": 1000},
}


def get_range(test_name: str, gender: str = "male") -> dict:
    key = test_name.lower().strip()
    ref = REFERENCE_RANGES.get(key)
    if not ref:
        return {}
    if "male" in ref and "female" in ref:
        return {**ref[gender.lower()], "unit": ref["unit"], "category": ref["category"],
                "meaning": ref["meaning"], "specialist": ref["specialist"]}
    return ref


def classify_value(test_name: str, value: float, gender: str = "male") -> dict:
    key = test_name.lower().strip()
    ref = get_range(key, gender)

    if not ref:
        return {"status": "unknown", "severity": "unknown",
                "message": "Reference range not available.", "specialist": None, "unit": ""}

    low = ref.get("min", 0)
    high = ref.get("max", 9999)
    unit = ref.get("unit", "")
    specialist = ref.get("specialist", "General Physician")
    meaning = ref.get("meaning", "")
    range_size = high - low

    panic = PANIC_VALUES.get(key, {})
    if panic:
        if "low" in panic and value < panic["low"]:
            return {"status": "LOW", "severity": "panic",
                    "message": f"🚨 PANIC VALUE — Seek emergency care immediately. {meaning}",
                    "specialist": specialist, "unit": unit}
        if "high" in panic and value > panic["high"]:
            return {"status": "HIGH", "severity": "panic",
                    "message": f"🚨 PANIC VALUE — Seek emergency care immediately. {meaning}",
                    "specialist": specialist, "unit": unit}

    if value < low:
        deviation = (low - value) / range_size if range_size > 0 else 1
        severity = "borderline" if deviation <= BORDERLINE_THRESHOLD else "critical"
        label = "🟡 BORDERLINE LOW" if severity == "borderline" else "🔴 LOW"
        return {"status": "LOW", "severity": severity,
                "message": f"{label} — {meaning}", "specialist": specialist, "unit": unit}

    elif value > high:
        deviation = (value - high) / range_size if range_size > 0 else 1
        severity = "borderline" if deviation <= BORDERLINE_THRESHOLD else "critical"
        label = "🟡 BORDERLINE HIGH" if severity == "borderline" else "🔴 HIGH"
        return {"status": "HIGH", "severity": severity,
                "message": f"{label} — {meaning}", "specialist": specialist, "unit": unit}

    return {"status": "NORMAL", "severity": "normal",
            "message": f"🟢 NORMAL. {meaning}", "specialist": None, "unit": unit}


def analyze_report(extracted_values: dict, gender: str = "male") -> list:
    results = []
    severity_order = {"panic": 0, "critical": 1, "borderline": 2, "normal": 3, "unknown": 4}

    for test_name, value in extracted_values.items():
        try:
            val = float(value)
        except (ValueError, TypeError):
            continue
        result = classify_value(test_name, val, gender)
        result["test_name"] = test_name
        result["value"] = val
        result["normal_range"] = get_range(test_name.lower(), gender)
        results.append(result)

    results.sort(key=lambda x: severity_order.get(x["severity"], 4))
    return results
