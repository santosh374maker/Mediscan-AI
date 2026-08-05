"""
PDF Extractor — extracts text and lab values from blood report PDFs.
Strategy 1: pdfplumber (digital PDFs)
Strategy 2: pytesseract OCR fallback (scanned/image PDFs)
"""
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)

ALIAS_MAP = {
    "hb": "haemoglobin", "hgb": "haemoglobin", "hemoglobin": "hemoglobin",
    "haemoglobin": "haemoglobin", "blood sugar fasting": "blood sugar fasting",
    "fasting blood sugar": "blood sugar fasting", "fasting glucose": "glucose fasting",
    "glucose fasting": "glucose fasting", "rbs": "postprandial glucose",
    "ppbs": "postprandial glucose", "hba1c": "hba1c",
    "glycated hemoglobin": "hba1c", "tsh": "tsh", "t3": "t3", "t4": "t4",
    "free t4": "free t4", "ft4": "free t4", "creatinine": "creatinine",
    "serum creatinine": "creatinine", "blood urea": "urea", "urea": "urea",
    "bun": "bun", "uric acid": "uric acid", "egfr": "egfr",
    "alt": "alt", "sgpt": "alt", "ast": "ast", "sgot": "ast",
    "alkaline phosphatase": "alkaline phosphatase", "alp": "alkaline phosphatase",
    "bilirubin": "bilirubin total", "total bilirubin": "bilirubin total",
    "albumin": "albumin", "total cholesterol": "total cholesterol",
    "cholesterol": "total cholesterol", "ldl": "ldl", "ldl cholesterol": "ldl",
    "hdl": "hdl", "hdl cholesterol": "hdl", "triglycerides": "triglycerides",
    "tg": "triglycerides", "wbc": "wbc", "total wbc": "wbc",
    "total leucocyte count": "wbc", "tlc": "wbc", "rbc": "rbc",
    "red blood cells": "rbc", "platelets": "platelets", "platelet count": "platelets",
    "plt": "platelets", "hematocrit": "hematocrit", "pcv": "hematocrit",
    "mcv": "mcv", "mch": "mch", "neutrophils": "neutrophils",
    "lymphocytes": "lymphocytes", "serum iron": "serum iron",
    "ferritin": "ferritin", "tibc": "tibc",
}

VALUE_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\(\)\/\-\.]{2,40}?)"
    r"[\s:|\-]{1,5}"
    r"(\d{1,4}(?:\.\d{1,3})?)"
    r"\s*"
    r"([a-zA-Z\/µ%³⁶°]+(?:\/[a-zA-Z³µL]+)?)?",
    re.MULTILINE
)


def extract_text_pdfplumber(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)
        return ""


def extract_text_ocr(pdf_path: str) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=300)
        return "\n".join(pytesseract.image_to_string(p, lang="eng") for p in pages)
    except Exception as e:
        logger.error("OCR failed: %s", e)
        return ""


def normalize_test_name(raw_name: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", raw_name.lower().strip())
    cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()
    if cleaned in ALIAS_MAP:
        return ALIAS_MAP[cleaned]
    for alias, canonical in ALIAS_MAP.items():
        if alias in cleaned:
            return canonical
    return None


def parse_values_from_text(text: str) -> dict:
    extracted = {}
    seen = set()
    for match in VALUE_PATTERN.finditer(text):
        raw_name = match.group(1).strip()
        raw_value = match.group(2).strip()
        canonical = normalize_test_name(raw_name)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        try:
            extracted[canonical] = float(raw_value)
        except ValueError:
            continue
    logger.info("Parsed %d lab values.", len(extracted))
    return extracted


def process_blood_report(pdf_path: str) -> dict:
    raw_text = extract_text_pdfplumber(pdf_path)
    method = "pdfplumber"
    if len(raw_text.strip()) < 100:
        raw_text = extract_text_ocr(pdf_path)
        method = "ocr"
    if not raw_text.strip():
        return {"raw_text": "", "extracted_values": {}, "extraction_method": "failed"}
    return {
        "raw_text": raw_text,
        "extracted_values": parse_values_from_text(raw_text),
        "extraction_method": method,
    }
