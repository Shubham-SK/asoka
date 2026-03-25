from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging import setup_logging
from app.slack.bolt_app import build_slack_app

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title="Salesforce Slackbot", version="0.1.0")

slack_handler = None
if settings.slack_enabled:
    _, slack_handler = build_slack_app(settings)


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
