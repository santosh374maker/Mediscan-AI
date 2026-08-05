from src.ml.router import route, route_with_coverage
from src.ml.predict import predict_all_panels
from src.ml.features import build_feature_vector, coverage
from src.ml.schemas import PANELS


def test_route_returns_metabolic_when_enough_features_present():
    values = {"glucose fasting": 140, "hba1c": 6.8, "triglycerides": 200}
    panels = route(values)
    assert "metabolic" in panels


def test_route_excludes_panel_below_coverage_threshold():
    values = {"glucose fasting": 90}  # only 1 of 6 metabolic features
    panels = route(values)
    assert "metabolic" not in panels


def test_route_with_coverage_reports_eligibility_detail():
    values = {"creatinine": 1.4, "egfr": 55}
    report = route_with_coverage(values)
    assert report["renal"]["eligible"] is True
    assert report["renal"]["covered_features"] == 2


def test_route_can_return_multiple_panels_at_once():
    values = {
        "glucose fasting": 140, "hba1c": 6.8, "triglycerides": 200,  # metabolic
        "creatinine": 1.5, "egfr": 50,  # renal
    }
    panels = route(values)
    assert "metabolic" in panels and "renal" in panels


def test_build_feature_vector_uses_nan_for_missing_tests():
    import numpy as np
    spec = PANELS["metabolic"]
    vec = build_feature_vector({"glucose fasting": 100}, spec.features, gender="male")
    assert len(vec) == len(spec.features)
    assert np.isnan(vec[1])  # hba1c not provided


def test_predict_all_panels_marks_ineligible_panel_with_none_probability():
    result = predict_all_panels({"glucose fasting": 100}, gender="male")
    assert result["liver"]["risk_probability"] is None
    assert result["liver"]["eligible"] is False
    assert result["liver"]["note"] is not None


def test_predict_all_panels_scores_eligible_panel():
    values = {"glucose fasting": 145, "hba1c": 6.8, "triglycerides": 210, "hdl": 34}
    result = predict_all_panels(values, gender="female")
    assert result["metabolic"]["eligible"] is True
    assert result["metabolic"]["risk_probability"] is not None
    assert 0.0 <= result["metabolic"]["risk_probability"] <= 1.0


def test_predict_all_panels_normal_values_score_low_risk():
    values = {"glucose fasting": 88, "hba1c": 5.1, "triglycerides": 95, "hdl": 62}
    result = predict_all_panels(values, gender="female")
    assert result["metabolic"]["risk_probability"] < 0.5
    assert result["metabolic"]["risk_label"] == "normal_pattern"
