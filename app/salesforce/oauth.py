from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode
from urllib.parse import urlparse

import requests

from app.config import Settings, get_settings
from app.db.repository import get_active_oauth_identity, upsert_oauth_identity
from app.db.session import SessionLocal
from app.security.crypto import decrypt_secret, encrypt_secret

STATE_TTL_SECONDS = 15 * 60
DEFAULT_TOKEN_TTL_SECONDS = 90 * 60


def build_oauth_start_url(
    slack_user_id: str,
    workspace_id: str,
    settings: Settings | None = None,
) -> str:
    cfg = settings or get_settings()
    base = cfg.app_base_url.rstrip("/")
    params = urlencode({"slack_user_id": slack_user_id, "workspace_id": workspace_id})
    return f"{base}/oauth/salesforce/start?{params}"


def build_salesforce_authorize_url(slack_user_id: str, workspace_id: str) -> str:
    settings = get_settings()
    _assert_oauth_configured(settings)
    state = _encode_state(
        {
            "slack_user_id": slack_user_id,
            "workspace_id": workspace_id,
            "ts": int(datetime.now(tz=UTC).timestamp()),
        }
    )
    params = urlencode(
        {
            "response_type": "code",
            "client_id": settings.salesforce_oauth_client_id,
            "redirect_uri": settings.salesforce_oauth_redirect_uri,
            "scope": "api refresh_token offline_access",
            "state": state,
        }
    )
    return f"{_oauth_host(settings)}/services/oauth2/authorize?{params}"


def handle_oauth_callback(code: str, state: str) -> dict[str, str]:
    settings = get_settings()
    _assert_oauth_configured(settings)
    payload = _decode_state(state)
    slack_user_id = str(payload["slack_user_id"])
    workspace_id = str(payload["workspace_id"])

    token_response = requests.post(
        f"{_oauth_host(settings)}/services/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "client_id": settings.salesforce_oauth_client_id,
            "client_secret": settings.salesforce_oauth_client_secret,
            "redirect_uri": settings.salesforce_oauth_redirect_uri,
            "code": code,
        },
        timeout=20,
    )
    token_response.raise_for_status()
    token_data = token_response.json()

    issued_at_raw = str(token_data.get("issued_at") or "")
    token_expires_at = _estimate_token_expiry(issued_at_raw)
    access_token = str(token_data["access_token"])
    refresh_token = token_data.get("refresh_token")
    instance_url = str(token_data["instance_url"])
    scope = token_data.get("scope")
    identity_url = token_data.get("id")

    sf_user_id = None
    sf_username = None
    sf_org_id = "default"
    if isinstance(identity_url, str) and identity_url:
        id_resp = requests.get(
            identity_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if id_resp.ok:
            id_data = id_resp.json()
            sf_user_id = id_data.get("user_id")
            sf_username = id_data.get("username")
            sf_org_id = id_data.get("organization_id") or "default"

    with SessionLocal() as db:
        upsert_oauth_identity(
            db=db,
            workspace_id=workspace_id,
            slack_user_id=slack_user_id,
            salesforce_org_key=str(sf_org_id),
            salesforce_user_id=str(sf_user_id) if sf_user_id else None,
            salesforce_username=str(sf_username) if sf_username else None,
            instance_url=instance_url,
            access_token_encrypted=encrypt_secret(access_token),
            refresh_token_encrypted=encrypt_secret(str(refresh_token)) if refresh_token else None,
            token_expires_at=token_expires_at,
            scopes=str(scope) if scope else None,
            metadata_json={"identity_url": identity_url} if identity_url else {},
        )
        db.commit()

    return {
        "slack_user_id": slack_user_id,
        "workspace_id": workspace_id,
        "salesforce_org_key": str(sf_org_id),
    }


def has_user_oauth_identity(slack_user_id: str, workspace_id: str) -> bool:
    with SessionLocal() as db:
        identity = get_active_oauth_identity(
            db=db,
            workspace_id=workspace_id,
            slack_user_id=slack_user_id,
        )
        return identity is not None


def get_user_oauth_session(slack_user_id: str, workspace_id: str) -> tuple[str, str] | None:
    with SessionLocal() as db:
        identity = get_active_oauth_identity(
            db=db,
            workspace_id=workspace_id,
            slack_user_id=slack_user_id,
        )
        if identity is None or not identity.instance_url or not identity.access_token_encrypted:
            return None

        if _is_token_expired(identity.token_expires_at):
            refreshed = _refresh_oauth_identity(identity.id)
            if refreshed is None:
                return None
            return refreshed

        return identity.instance_url, decrypt_secret(identity.access_token_encrypted)


def _refresh_oauth_identity(identity_id: str) -> tuple[str, str] | None:
    settings = get_settings()
    with SessionLocal() as db:
        from app.db.models import SlackSalesforceIdentity  # local import to avoid cycles

        identity = db.get(SlackSalesforceIdentity, identity_id)
        if identity is None or not identity.refresh_token_encrypted:
            return None
        refresh_token = decrypt_secret(identity.refresh_token_encrypted)

        response = requests.post(
            f"{_oauth_host(settings)}/services/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "client_id": settings.salesforce_oauth_client_id,
                "client_secret": settings.salesforce_oauth_client_secret,
                "refresh_token": refresh_token,
            },
            timeout=20,
        )
        if not response.ok:
            return None
        token_data = response.json()
        new_access_token = str(token_data["access_token"])
        instance_url = str(token_data.get("instance_url") or identity.instance_url or "")
        if not instance_url:
            return None
        identity.access_token_encrypted = encrypt_secret(new_access_token)
        identity.instance_url = instance_url
        identity.token_expires_at = datetime.now(tz=UTC) + timedelta(seconds=DEFAULT_TOKEN_TTL_SECONDS)
        db.commit()
        return instance_url, new_access_token


def _assert_oauth_configured(settings: Settings) -> None:
    if not settings.salesforce_oauth_enabled:
        raise RuntimeError(
            "Salesforce OAuth is not configured. Set SALESFORCE_OAUTH_CLIENT_ID, "
            "SALESFORCE_OAUTH_CLIENT_SECRET, and SALESFORCE_OAUTH_REDIRECT_URI."
        )


def _oauth_host(settings: Settings) -> str:
    raw_domain = (settings.salesforce_domain or "login").strip()
    domain = raw_domain.lower()
    if domain.startswith("http://") or domain.startswith("https://"):
        parsed = urlparse(raw_domain)
        if parsed.hostname:
            return f"{parsed.scheme or 'https'}://{parsed.hostname}"
    if "." in domain and "/" not in domain:
        return f"https://{domain}"
    if domain == "test":
        return "https://test.salesforce.com"
    return "https://login.salesforce.com"


def _state_secret() -> bytes:
    settings = get_settings()
    secret = settings.oauth_state_secret.strip() or settings.slack_signing_secret.strip()
    if not secret:
        raise RuntimeError("Set OAUTH_STATE_SECRET (or SLACK_SIGNING_SECRET) for OAuth state signing.")
    return secret.encode("utf-8")


def _encode_state(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    sig = hmac.new(_state_secret(), raw, hashlib.sha256).digest()
    blob = base64.urlsafe_b64encode(raw + b"." + sig).decode("utf-8")
    return blob.rstrip("=")


def _decode_state(state: str) -> dict[str, object]:
    padded = state + "=" * (-len(state) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
    raw, sig = decoded.rsplit(b".", 1)
    expected = hmac.new(_state_secret(), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise RuntimeError("Invalid OAuth state signature.")
    payload = json.loads(raw.decode("utf-8"))
    ts = int(payload.get("ts", 0))
    age_seconds = int(datetime.now(tz=UTC).timestamp()) - ts
    if age_seconds < 0 or age_seconds > STATE_TTL_SECONDS:
        raise RuntimeError("OAuth state expired. Please restart connect flow.")
    return payload


def _estimate_token_expiry(issued_at_millis: str) -> datetime:
    try:
        issued_ms = int(issued_at_millis)
        return datetime.fromtimestamp(issued_ms / 1000, tz=UTC) + timedelta(
            seconds=DEFAULT_TOKEN_TTL_SECONDS
        )
    except ValueError:
        return datetime.now(tz=UTC) + timedelta(seconds=DEFAULT_TOKEN_TTL_SECONDS)


def _is_token_expired(token_expires_at: datetime | None) -> bool:
    if token_expires_at is None:
        return False
    now_utc = datetime.now(tz=UTC)
    expires_utc = _to_utc_aware(token_expires_at)
    return expires_utc <= now_utc + timedelta(minutes=5)


def _to_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        # SQLite often returns naive datetimes even when original values were UTC.
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
