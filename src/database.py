"""
Database layer — SQLite via SQLAlchemy.
Defines User and Report models.
"""
import logging
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

logger = logging.getLogger(__name__)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    extracted_values = Column(Text, nullable=True)   # JSON string
    analysis_result = Column(Text, nullable=True)    # JSON string
    ai_explanation = Column(Text, nullable=True)     # Plain-English explanation
    severity_score = Column(String(20), nullable=True)
    report_path = Column(String(500), nullable=True)
    ml_risk_predictions = Column(Text, nullable=True)  # JSON string — specialist model outputs


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
