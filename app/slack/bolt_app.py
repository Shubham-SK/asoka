from __future__ import annotations

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler

from app.config import Settings
from app.slack.handlers import register_handlers


def build_slack_app(settings: Settings) -> tuple[App, SlackRequestHandler]:
    slack_app = App(token=settings.slack_bot_token, signing_secret=settings.slack_signing_secret)
    register_handlers(slack_app, settings)
    return slack_app, SlackRequestHandler(slack_app)
