"""
Panel definitions for the specialist risk models.

Each panel maps to a real clinical test grouping (matching the `category`
field already used in src/reference_ranges.py) so the ML layer sits on top
of, rather than duplicates, the existing rule-based logic.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PanelSpec:
    name: str
    label: str
    features: List[str]              # canonical test names (match reference_ranges.py keys)
    min_features_required: int       # coverage gate — below this, don't run the model
    positive_class_name: str         # human-readable label for class 1
    description: str


PANELS: Dict[str, PanelSpec] = {
    "metabolic": PanelSpec(
        name="metabolic",
        label="Metabolic / Diabetes Risk",
        features=["glucose fasting", "hba1c", "triglycerides", "hdl", "ldl", "total cholesterol"],
        min_features_required=3,
        positive_class_name="elevated_metabolic_risk",
        description="Diabetes / metabolic syndrome risk pattern from sugar + lipid panel.",
    ),
    "hematology": PanelSpec(
        name="hematology",
        label="CBC / Hematology Risk",
        features=["haemoglobin", "wbc", "rbc", "platelets", "hematocrit", "mcv", "mch"],
        min_features_required=3,
        positive_class_name="elevated_hematologic_risk",
        description="Anaemia / infection / marrow-related risk pattern from CBC panel.",
    ),
    "liver": PanelSpec(
        name="liver",
        label="Liver Function Risk",
        features=["alt", "ast", "alkaline phosphatase", "bilirubin total", "albumin"],
        min_features_required=3,
        positive_class_name="elevated_liver_risk",
        description="Hepatocellular / cholestatic risk pattern from LFT panel.",
    ),
    "renal": PanelSpec(
        name="renal",
        label="Kidney / Renal Risk",
        features=["creatinine", "urea", "bun", "uric acid", "egfr"],
        min_features_required=2,
        positive_class_name="elevated_renal_risk",
        description="Renal impairment risk pattern from kidney function panel.",
    ),
}


def get_panel_feature_map() -> Dict[str, List[str]]:
    return {name: spec.features for name, spec in PANELS.items()}
