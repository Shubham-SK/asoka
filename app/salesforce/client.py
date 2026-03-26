from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from simple_salesforce import Salesforce

from app.config import get_settings
from app.salesforce.oauth import get_user_oauth_session


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
def _get_integration_salesforce_client() -> Salesforce:
    settings = get_settings()
    if not settings.salesforce_enabled:
        raise RuntimeError("Salesforce credentials are not configured.")

    return Salesforce(
        username=settings.salesforce_username,
        password=settings.salesforce_password,
        security_token=settings.salesforce_security_token,
        domain=_normalize_domain(settings.salesforce_domain),
    )


def get_salesforce_client(
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> Salesforce:
    if slack_user_id and workspace_id:
        oauth_session = get_user_oauth_session(slack_user_id=slack_user_id, workspace_id=workspace_id)
        if oauth_session is not None:
            instance_url, access_token = oauth_session
            return Salesforce(instance_url=instance_url, session_id=access_token)

    settings = get_settings()
    if settings.salesforce_enabled:
        return _get_integration_salesforce_client()

    raise RuntimeError(
        "No Salesforce auth available for this user. Connect OAuth at "
        "/oauth/salesforce/start or configure integration credentials."
    )
