from __future__ import annotations

from anthropic import Anthropic

from app.config import get_settings


def get_claude_client() -> Anthropic:
    settings = get_settings()
    if not settings.llm_enabled:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    return Anthropic(api_key=settings.anthropic_api_key)
