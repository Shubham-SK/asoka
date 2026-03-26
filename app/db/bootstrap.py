from __future__ import annotations

from app.db.base import Base
from app.db.session import get_engine


def init_db() -> None:
    # Ensure models are imported so SQLAlchemy can register all tables.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())
