from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings
from app.db.bootstrap import init_db
from app.logging import setup_logging
from app.salesforce.oauth import build_salesforce_authorize_url, handle_oauth_callback
from app.slack.bolt_app import build_slack_app

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title="Salesforce Slackbot", version="0.1.0")

slack_handler = None
if settings.slack_enabled:
    _, slack_handler = build_slack_app(settings)


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/slack/events")
async def slack_events(req: Request):
    if not settings.slack_enabled or slack_handler is None:
        raise HTTPException(
            status_code=503, detail="Slack is not configured. Set SLACK_BOT_TOKEN and secret."
        )
    return await slack_handler.handle(req)


@app.get("/oauth/salesforce/start")
async def oauth_salesforce_start(
    slack_user_id: str = Query(...),
    workspace_id: str = Query(...),
):
    try:
        url = build_salesforce_authorize_url(
            slack_user_id=slack_user_id,
            workspace_id=workspace_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=url, status_code=302)


@app.get("/oauth/salesforce/callback")
async def oauth_salesforce_callback(
    code: str = Query(...),
    state: str = Query(...),
):
    try:
        payload = handle_oauth_callback(code=code, state=state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {exc}") from exc
    return JSONResponse(
        {
            "status": "connected",
            "message": "Salesforce OAuth connected. You can return to Slack and retry your request.",
            "slack_user_id": payload["slack_user_id"],
            "workspace_id": payload["workspace_id"],
            "salesforce_org_key": payload["salesforce_org_key"],
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "sf-slackbot",
            "phase": "0-1",
            "slack_enabled": settings.slack_enabled,
            "salesforce_enabled": settings.salesforce_enabled,
        }
    )
