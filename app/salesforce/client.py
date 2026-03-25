from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from simple_salesforce import Salesforce

from app.config import get_settings


def _normalize_domain(raw_domain: str) -> str:
    """
    simple-salesforce expects a domain key like "login" or "test",
    not a full URL/hostname. Accept common forms and normalize safely.
    """
    value = (raw_domain or "").strip().lower()
    if value in {"login", "test"}:
        return value

    if value.startswith("http://") or value.startswith("https://"):
        value = (urlparse(value).hostname or "").lower()

    if "sandbox" in value or ".test." in value:
        return "test"

    # Default to login for custom/instance hostnames.
    return "login"


@lru_cache
def get_salesforce_client() -> Salesforce:
    settings = get_settings()
    if not settings.salesforce_enabled:
        raise RuntimeError("Salesforce credentials are not configured.")

    return Salesforce(
        username=settings.salesforce_username,
        password=settings.salesforce_password,
        security_token=settings.salesforce_security_token,
        domain=_normalize_domain(settings.salesforce_domain),
    )
