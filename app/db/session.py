from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _default_sqlite_url() -> str:
    # Local default for early development; replace with Postgres in deployment.
    return "sqlite:///./app.db"


def get_engine():
    settings = get_settings()
    db_url = getattr(settings, "database_url", "") or _default_sqlite_url()
    return create_engine(db_url, future=True)


SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, class_=Session)
