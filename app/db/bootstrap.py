from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy import text

from app.db.base import Base
from app.db.session import get_engine


def init_db() -> None:
    # Ensure models are imported so SQLAlchemy can register all tables.
    from app.db import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_runtime_columns(engine)


def _ensure_runtime_columns(engine) -> None:  # noqa: ANN001
    inspector = inspect(engine)
    if not inspector.has_table("knowledge_items"):
        return
    columns = {col.get("name") for col in inspector.get_columns("knowledge_items")}
    if "usage_count" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE knowledge_items "
                    "ADD COLUMN usage_count INTEGER NOT NULL DEFAULT 0"
                )
            )
