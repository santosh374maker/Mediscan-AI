import pytest
from pydantic import ValidationError

from src.schemas import LLMExtractionResult, reconcile_with_confidence


def test_llm_extraction_result_validates_correct_shape():
    result = LLMExtractionResult.model_validate({
        "values": [{"test_name": "HbA1c", "value": 6.9, "unit": "%"}]
    })
    assert result.as_dict() == {"hba1c": 6.9}


def test_llm_extraction_result_normalizes_test_name_case():
    result = LLMExtractionResult.model_validate({
        "values": [{"test_name": "  Glucose Fasting  ", "value": 90}]
    })
    assert "glucose fasting" in result.as_dict()


def test_llm_extraction_result_rejects_non_numeric_value():
    with pytest.raises(ValidationError):
        LLMExtractionResult.model_validate({
            "values": [{"test_name": "glucose", "value": "high"}]
        })


def test_llm_extraction_result_defaults_to_empty_list():
    result = LLMExtractionResult.model_validate({})
    assert result.as_dict() == {}


def test_reconcile_agreement_gives_high_confidence():
    records = reconcile_with_confidence({"glucose fasting": 100.0}, {"glucose fasting": 100.0})
    assert len(records) == 1
    assert records[0].confidence == "high"
    assert set(records[0].sources) == {"regex", "llm"}


def test_reconcile_disagreement_flagged_as_conflict():
    records = reconcile_with_confidence({"glucose fasting": 100.0}, {"glucose fasting": 250.0})
    assert records[0].confidence == "conflict"
    assert records[0].conflicting_value == 250.0


def test_reconcile_regex_only_value_is_medium_confidence():
    records = reconcile_with_confidence({"glucose fasting": 100.0}, {})
    assert records[0].confidence == "medium"
    assert records[0].sources == ["regex"]


def test_reconcile_llm_only_value_is_medium_confidence():
    records = reconcile_with_confidence({}, {"hba1c": 6.1})
    assert records[0].confidence == "medium"
    assert records[0].sources == ["llm"]
