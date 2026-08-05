import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sample_report.pdf"
    c = canvas.Canvas(str(path))
    lines = [
        "DIABETES PANEL",
        "Fasting Glucose: 142 mg/dL",
        "HbA1c - 6.8 %",
        "LIPID PROFILE",
        "Total Cholesterol : 230 mg/dL",
        "HDL Cholesterol: 32 mg/dL",
        "Triglycerides 220 mg/dL",
    ]
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
    c.save()
    return path


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Isolated DB for this test module so it doesn't collide with dev data.
    test_dir = tmp_path_factory.mktemp("app_data")
    os.environ["DATABASE_URL"] = f"sqlite:///{test_dir}/test_app.db"

    import src.llm as llm_module
    def fake_call_llm(prompt, history=None, system="", temperature=0.3, response_format=None, use_cache=True):
        return "Mocked AI explanation for testing."
    llm_module.call_llm = fake_call_llm

    import src.api as api_module
    api_module.call_llm = fake_call_llm

    with TestClient(api_module.app) as c:
        yield c


def test_health_check(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "rag_engine" in body
    assert "llm_cache" in body


def test_signup_returns_access_token(client):
    r = client.post("/auth/signup", json={
        "username": "integration_user", "email": "integration@example.com", "password": "testpass123",
    })
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_signup_duplicate_username_rejected(client):
    client.post("/auth/signup", json={
        "username": "dup_user", "email": "dup1@example.com", "password": "testpass123",
    })
    r = client.post("/auth/signup", json={
        "username": "dup_user", "email": "dup2@example.com", "password": "testpass123",
    })
    assert r.status_code == 400


def test_upload_requires_auth(client, sample_pdf):
    with open(sample_pdf, "rb") as f:
        r = client.post("/report/upload", files={"file": ("r.pdf", f, "application/pdf")})
    assert r.status_code in (401, 403)


def test_upload_report_end_to_end(client, sample_pdf):
    signup = client.post("/auth/signup", json={
        "username": "upload_user", "email": "upload@example.com", "password": "testpass123",
    })
    token = signup.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open(sample_pdf, "rb") as f:
        r = client.post(
            "/report/upload",
            files={"file": ("sample_report.pdf", f, "application/pdf")},
            params={"gender": "female"},
            headers=headers,
        )
    assert r.status_code == 200
    data = r.json()

    assert "extracted_values" in data
    assert data["extracted_values"].get("glucose fasting") == 142.0

    assert "ml_risk_predictions" in data
    assert "metabolic" in data["ml_risk_predictions"]
    assert data["ml_risk_predictions"]["metabolic"]["eligible"] is True
    assert data["ml_risk_predictions"]["metabolic"]["risk_probability"] > 0.5

    assert "disclaimer" in data
    assert data["disclaimer"]["level"] in ("safe", "warning", "critical", "emergency")


def test_invalid_token_rejected(client):
    r = client.get("/report/history", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401
