"""
MediScan AI — FastAPI backend
Features: Auth, blood report upload, AI analysis, streaming explanation,
          report history, PDF download, analytics, ML risk scoring.
"""
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src import analytics
from src.auth import (authenticate_user, create_access_token, create_user,
                       get_current_user, get_user_by_email, get_user_by_username,
                       require_admin, seed_admin_user)
from src.cache import cache_stats
from src.config import (CORS_ORIGINS, RATE_LIMIT_ASK, RATE_LIMIT_UPLOAD,
                         REPORTS_DIR, validate_production_security)
from src.database import Report, SessionLocal, get_db, init_db
from src.disclaimer import generate_disclaimer
from src.llm import call_llm, stream_llm
from src.pdf_extractor import process_blood_report
from src.llm_extractor import extract_with_fallback
from src.ml.predict import predict_all_panels
from src.rag_engine import rag_engine
from src.reference_ranges import analyze_report
from src.report_generator import generate_pdf_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MediScan AI API",
    description="AI-powered blood report analysis",
    version="1.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Schemas ───────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str = "user"


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    report_id: int = None


# ── Startup ───────────────────────────────────────────

@app.on_event("startup")
def startup():
    validate_production_security()
    logger.info("Initializing database...")
    init_db()
    db = SessionLocal()
    try:
        seed_admin_user(db)
    finally:
        db.close()
    logger.info("Loading RAG engine...")
    rag_engine.load()
    logger.info("MediScan AI ready ✅")


# ── Health ────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {
        "status": "MediScan AI running 🩺",
        "rag_engine": rag_engine.status(),
        "llm_cache": cache_stats(),
    }


# ── Auth ──────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password")
    return {"access_token": create_access_token({"sub": user.username}), "role": user.role}


@app.post("/auth/signup", response_model=TokenResponse, tags=["Auth"])
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    if get_user_by_username(db, payload.username):
        raise HTTPException(status_code=400, detail="Username already taken.")
    if get_user_by_email(db, payload.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    user = create_user(db, payload.username, payload.email, payload.password)
    return {"access_token": create_access_token({"sub": user.username}), "role": user.role}


# ── Upload & Analyze ──────────────────────────────────

@app.post("/report/upload", tags=["Reports"])
@limiter.limit(RATE_LIMIT_UPLOAD)
def upload_report(
    request: Request,
    file: UploadFile = File(...),
    gender: str = "male",
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    t0 = time.time()
    save_path = REPORTS_DIR / f"{uuid.uuid4().hex}_{file.filename}"

    try:
        content = file.file.read()
        if len(content) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

        with open(save_path, "wb") as f:
            f.write(content)

        # Step 1: Extract text + regex parse
        extraction = process_blood_report(str(save_path))
        raw_text = extraction["raw_text"]
        regex_values = extraction["extracted_values"]

        if not raw_text:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        # Step 2: LLM extraction to fill gaps
        extracted_values = extract_with_fallback(raw_text, regex_values)

        if not extracted_values:
            raise HTTPException(status_code=422,
                                detail="No lab values could be extracted from this report.")

        # Step 3: Compare with reference ranges (existing rule-based severity)
        analysis_results = analyze_report(extracted_values, gender=gender)

        # Step 3b: Specialist ML risk models (secondary signal, not a diagnosis).
        # Router decides which of the 4 panel models have enough feature
        # coverage to run at all — see src/ml/router.py.
        ml_predictions = predict_all_panels(extracted_values, gender=gender)
        ml_risk_summary_lines = [
            f"- {p['panel_label']}: {p['risk_label']} (probability={p['risk_probability']})"
            for p in ml_predictions.values() if p["risk_probability"] is not None
        ]
        ml_risk_summary = "\n".join(ml_risk_summary_lines) if ml_risk_summary_lines else \
            "No panel had enough matching tests for the ML risk models to run."

        # Step 4: Generate disclaimer
        disclaimer = generate_disclaimer(analysis_results)

        # Step 5: RAG-powered AI explanation
        rag_results = rag_engine.retrieve(
            f"Explain these blood test results: {', '.join(extracted_values.keys())}"
        )
        rag_context = rag_engine.build_context(rag_results)

        abnormal = [r for r in analysis_results if r["severity"] != "normal"]
        abnormal_summary = "\n".join(
            f"- {r['test_name']}: {r['value']} {r['unit']} ({r['status']})"
            for r in abnormal[:10]
        )

        prompt = f"""You are a medical AI assistant explaining blood report results to a patient in simple language.

Patient's abnormal values:
{abnormal_summary if abnormal_summary else "All values within normal range."}

Secondary ML risk-pattern signals (NOT a diagnosis — a supporting signal only):
{ml_risk_summary}

Medical context:
{rag_context}

Explain what these results mean, why they might be abnormal, and what the patient should do next.
Use simple, reassuring language. Do not diagnose. Always recommend consulting a doctor.
Keep response under 300 words."""

        ai_explanation = call_llm(prompt)

        # Step 6: Final disclaimer update with AI text
        disclaimer = generate_disclaimer(analysis_results, ai_text=ai_explanation)

        # Step 7: Save to DB
        report = Report(
            user_id=current_user.id,
            filename=file.filename,
            report_path=str(save_path),
            extracted_values=json.dumps(extracted_values),
            analysis_result=json.dumps(analysis_results),
            ai_explanation=ai_explanation,
            severity_score=disclaimer["level"],
            ml_risk_predictions=json.dumps(ml_predictions),
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        latency = (time.time() - t0) * 1000
        analytics.log_query(
            question=f"Upload: {file.filename}",
            answer=ai_explanation,
            sources=[r["title"] for r in rag_results],
            user=current_user.username,
            latency_ms=latency,
        )

        return {
            "report_id": report.id,
            "extracted_values": extracted_values,
            "analysis_results": analysis_results,
            "ml_risk_predictions": ml_predictions,
            "ai_explanation": ai_explanation,
            "disclaimer": disclaimer,
            "extraction_method": extraction["extraction_method"],
            "latency_ms": round(latency, 1),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report processing error: %s", e)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ── Follow-up Questions ───────────────────────────────

@app.post("/report/ask", tags=["Reports"])
@limiter.limit(RATE_LIMIT_ASK)
def ask_about_report(
    request: Request,
    q: QuestionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Ask a follow-up question about your blood report."""
    context = ""
    if q.report_id:
        report = db.query(Report).filter(
            Report.id == q.report_id,
            Report.user_id == current_user.id
        ).first()
        if report and report.analysis_result:
            results = json.loads(report.analysis_result)
            abnormal = [r for r in results if r["severity"] != "normal"]
            context = "Patient's abnormal values:\n" + "\n".join(
                f"- {r['test_name']}: {r['value']} ({r['status']})"
                for r in abnormal[:10]
            )

    rag_results = rag_engine.retrieve(q.question)
    rag_context = rag_engine.build_context(rag_results)

    prompt = f"""You are a medical AI assistant. Answer the patient's question clearly and simply.
Always recommend consulting a doctor for medical decisions.

{context}

Medical knowledge:
{rag_context}

Question: {q.question}

Answer in simple, plain English:"""

    t0 = time.time()
    answer = call_llm(prompt)
    latency = (time.time() - t0) * 1000

    analytics.log_query(q.question, answer, [r["title"] for r in rag_results],
                        current_user.username, latency)

    return {"answer": answer, "latency_ms": round(latency, 1)}


@app.post("/report/ask/stream", tags=["Reports"])
@limiter.limit(RATE_LIMIT_ASK)
def ask_stream(
    request: Request,
    q: QuestionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Streaming follow-up question about your report."""
    context = ""
    if q.report_id:
        report = db.query(Report).filter(
            Report.id == q.report_id,
            Report.user_id == current_user.id
        ).first()
        if report and report.analysis_result:
            results = json.loads(report.analysis_result)
            abnormal = [r for r in results if r["severity"] != "normal"]
            context = "Patient's abnormal values:\n" + "\n".join(
                f"- {r['test_name']}: {r['value']} ({r['status']})"
                for r in abnormal[:10]
            )

    rag_results = rag_engine.retrieve(q.question)
    rag_context = rag_engine.build_context(rag_results)

    prompt = f"""You are a medical AI assistant. Answer clearly and simply.
Always recommend consulting a doctor for medical decisions.

{context}

Medical knowledge:
{rag_context}

Question: {q.question}

Answer in simple, plain English:"""

    def generate():
        for token in stream_llm(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Report History ────────────────────────────────────

@app.get("/report/history", tags=["Reports"])
def get_report_history(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    reports = db.query(Report).filter(
        Report.user_id == current_user.id
    ).order_by(Report.uploaded_at.desc()).limit(20).all()

    return [
        {
            "id": r.id,
            "filename": r.filename,
            "uploaded_at": r.uploaded_at.isoformat(),
            "severity_score": r.severity_score,
        }
        for r in reports
    ]


@app.get("/report/{report_id}", tags=["Reports"])
def get_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    return {
        "id": report.id,
        "filename": report.filename,
        "uploaded_at": report.uploaded_at.isoformat(),
        "extracted_values": json.loads(report.extracted_values or "{}"),
        "analysis_results": json.loads(report.analysis_result or "[]"),
        "ml_risk_predictions": json.loads(report.ml_risk_predictions or "{}"),
        "ai_explanation": report.ai_explanation,
        "severity_score": report.severity_score,
    }


# ── PDF Download ──────────────────────────────────────

@app.get("/report/{report_id}/download", tags=["Reports"])
def download_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    analysis_results = json.loads(report.analysis_result or "[]")
    disclaimer = generate_disclaimer(analysis_results, report.ai_explanation or "")

    pdf_bytes = generate_pdf_report(
        username=current_user.username,
        filename=report.filename,
        analysis_results=analysis_results,
        ai_explanation=report.ai_explanation or "",
        disclaimer=disclaimer,
    )

    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=mediscan_report_{report_id}.pdf"},
    )


# ── Analytics ─────────────────────────────────────────

@app.get("/analytics", tags=["Analytics"])
def get_analytics(current_user=Depends(require_admin)):
    return analytics.get_stats()
