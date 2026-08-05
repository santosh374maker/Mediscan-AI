"""
Synthetic training data generator.

No public lab-panel dataset is reachable from this environment, so training
data is generated synthetically and labeled with a *silver-labeling* /
rule-distillation approach: each patient's label is derived from established
multi-criteria clinical rules (e.g. ATP III-style metabolic syndrome
criteria — 2+ of {high glucose, high triglycerides, low HDL} = elevated
risk), plus Gaussian noise on the underlying values and a small amount of
explicit label noise to avoid the model trivially re-deriving the rule.

This is stated explicitly (not hidden) because it's the honest thing to do,
and because "distilling a known rule combination into a learned model with a
noisy/partial feature view" is itself a legitimate, explainable technique —
the value isn't the rule (we already have it in reference_ranges.py), it's
that the learned model generalizes under missing features and can be scored
in the same pipeline as everything else.

Swap `generate_*` for a real cohort dataset (e.g. NHANES, once available)
without touching train.py — the contract is just "DataFrame of feature
columns + `label`".
"""
import numpy as np
import pandas as pd

from src.ml.schemas import PANELS

RNG = np.random.default_rng(42)


def _inject_missingness(df: pd.DataFrame, feature_cols, missing_rate: float = 0.15) -> pd.DataFrame:
    """Randomly null out cells to mimic partial real-world panels."""
    mask = RNG.random(df[feature_cols].shape) < missing_rate
    df.loc[:, feature_cols] = df[feature_cols].mask(mask)
    return df


def _flip_label_noise(labels: np.ndarray, noise_rate: float = 0.05) -> np.ndarray:
    flip_mask = RNG.random(len(labels)) < noise_rate
    return np.where(flip_mask, 1 - labels, labels)


def _random_gender(n: int) -> np.ndarray:
    return RNG.choice(["male", "female"], size=n)


def generate_metabolic(n: int = 4000) -> pd.DataFrame:
    features = PANELS["metabolic"].features
    gender = _random_gender(n)
    glucose = RNG.normal(95, 20, n).clip(60, 300)
    hba1c = RNG.normal(5.4, 0.9, n).clip(4.0, 12.0)
    trig = RNG.normal(120, 60, n).clip(30, 600)
    hdl = RNG.normal(48, 14, n).clip(20, 100)
    ldl = RNG.normal(105, 35, n).clip(30, 250)
    chol = ldl + hdl + trig / 5 + RNG.normal(10, 8, n)

    hdl_threshold = np.where(gender == "male", 40, 50)
    criteria = (
        (glucose >= 100).astype(int)
        + (trig >= 150).astype(int)
        + (hdl < hdl_threshold).astype(int)
        + (hba1c >= 5.7).astype(int)
    )
    label = (criteria >= 2).astype(int)
    label = _flip_label_noise(label)

    df = pd.DataFrame({
        "gender": gender, "glucose fasting": glucose, "hba1c": hba1c, "triglycerides": trig,
        "hdl": hdl, "ldl": ldl, "total cholesterol": chol, "label": label,
    })
    return _inject_missingness(df, features)


def generate_hematology(n: int = 4000) -> pd.DataFrame:
    features = PANELS["hematology"].features
    gender = _random_gender(n)
    hb = RNG.normal(13.5, 2.2, n).clip(5, 19)
    wbc = RNG.normal(7.5, 3.0, n).clip(1.5, 25)
    rbc = RNG.normal(4.8, 0.7, n).clip(2.5, 7)
    platelets = RNG.normal(260, 90, n).clip(20, 700)
    hct = hb * 3 + RNG.normal(0, 2, n)
    mcv = RNG.normal(90, 10, n).clip(60, 120)
    mch = RNG.normal(30, 3, n).clip(20, 40)

    wbc_abnormal = (wbc > 12) | (wbc < 3.5)
    criteria = (
        (hb < 11.0).astype(int)
        + wbc_abnormal.astype(int)
        + (platelets < 100).astype(int)
        + (mcv < 75).astype(int)
    )
    label = (criteria >= 2).astype(int)
    label = _flip_label_noise(label)

    df = pd.DataFrame({
        "gender": gender, "haemoglobin": hb, "wbc": wbc, "rbc": rbc, "platelets": platelets,
        "hematocrit": hct, "mcv": mcv, "mch": mch, "label": label,
    })
    return _inject_missingness(df, features)


def generate_liver(n: int = 4000) -> pd.DataFrame:
    features = PANELS["liver"].features
    gender = _random_gender(n)
    alt = RNG.lognormal(3.4, 0.5, n).clip(5, 500)
    ast = RNG.lognormal(3.2, 0.5, n).clip(5, 500)
    alp = RNG.normal(90, 35, n).clip(20, 400)
    bilirubin = RNG.lognormal(-1.1, 0.6, n).clip(0.05, 10)
    albumin = RNG.normal(4.3, 0.6, n).clip(1.5, 5.8)

    criteria = (
        (alt > 56).astype(int)
        + (ast > 40).astype(int)
        + (bilirubin > 1.2).astype(int)
        + (albumin < 3.4).astype(int)
    )
    label = (criteria >= 2).astype(int)
    label = _flip_label_noise(label)

    df = pd.DataFrame({
        "gender": gender, "alt": alt, "ast": ast, "alkaline phosphatase": alp,
        "bilirubin total": bilirubin, "albumin": albumin, "label": label,
    })
    return _inject_missingness(df, features)


def generate_renal(n: int = 4000) -> pd.DataFrame:
    features = PANELS["renal"].features
    gender = _random_gender(n)
    creatinine = RNG.lognormal(-0.2, 0.35, n).clip(0.3, 8)
    urea = RNG.normal(15, 8, n).clip(3, 90)
    bun = urea * RNG.normal(1.0, 0.05, n)
    uric_acid = RNG.normal(5.2, 1.5, n).clip(1.5, 12)
    egfr = (120 - creatinine * 18).clip(5, 130) + RNG.normal(0, 8, n)

    criteria = (
        (creatinine > 1.2).astype(int)
        + (egfr < 60).astype(int)
        + (urea > 20).astype(int)
        + (uric_acid > 7.0).astype(int)
    )
    label = (criteria >= 2).astype(int)
    label = _flip_label_noise(label)

    df = pd.DataFrame({
        "gender": gender, "creatinine": creatinine, "urea": urea, "bun": bun,
        "uric acid": uric_acid, "egfr": egfr, "label": label,
    })
    return _inject_missingness(df, features)


GENERATORS = {
    "metabolic": generate_metabolic,
    "hematology": generate_hematology,
    "liver": generate_liver,
    "renal": generate_renal,
}
