from __future__ import annotations

import logging

from app.config import Settings
from app.db.repository import (
    append_conversation_message,
    load_conversation_window,
    set_user_context_entry,
)
from app.db.session import SessionLocal
from app.orchestrator.service import Orchestrator
from app.salesforce.oauth import build_oauth_start_url

logger = logging.getLogger(__name__)


def register_handlers(slack_app, settings: Settings) -> None:
    orchestrator = Orchestrator(settings)

    @slack_app.event("message")
    def handle_message_events(body, say, event, client, logger):  # noqa: ANN001
        # Ignore bot messages and non-DM events.
        if event.get("subtype") == "bot_message":
            return
        if event.get("channel_type") != "im":
            return

        text = event.get("text", "").strip()
        user_id = event.get("user", "")
        workspace_id = str(body.get("team_id") or event.get("team") or "default")

        if not text or not user_id:
            logger.info("Skipping empty DM event")
            return

        channel_id = event.get("channel", "")
        conversation_window = _load_dm_conversation_window_from_db(
            workspace_id=workspace_id,
            slack_user_id=user_id,
            channel_id=channel_id,
            limit=25,
        )
        _append_message_to_db(
            workspace_id=workspace_id,
            slack_user_id=user_id,
            channel_id=channel_id,
            role="user",
            text=text,
            slack_ts=str(event.get("ts") or ""),
        )
        _persist_dm_context(
            workspace_id=workspace_id,
            slack_user_id=user_id,
            conversation_window=conversation_window,
            last_user_message=text,
        )
        logger.info("Received DM from user=%s text=%s", user_id, text)
        normalized = text.lower()
        if normalized in {
            "connect salesforce",
            "salesforce connect",
            "connect sf",
            "oauth salesforce",
            "login salesforce",
        }:
            connect_url = build_oauth_start_url(
                slack_user_id=user_id,
                workspace_id=workspace_id,
                settings=settings,
            )
            connect_text = f"Connect your Salesforce account here:\n{connect_url}"
            sent = say(text=connect_text)
            _append_message_to_db(
                workspace_id=workspace_id,
                slack_user_id=user_id,
                channel_id=channel_id,
                role="assistant",
                text=connect_text,
                slack_ts=str(getattr(sent, "get", lambda *_: "")("ts") or ""),
            )
            return

        response_ts = ""
        if channel_id:
            try:
                posted = client.chat_postMessage(
                    channel=channel_id,
                    text="Working on it... I will post step-by-step updates here.",
                )
                response_ts = str(posted.get("ts", "") or "")
            except Exception as exc:
                logger.info("Could not post streaming placeholder message: %s", exc)

        def progress_update(update_text: str) -> None:
            if not channel_id or not response_ts:
                return
            try:
                client.chat_update(channel=channel_id, ts=response_ts, text=update_text)
            except Exception as exc:
                logger.info("Could not update streaming progress message: %s", exc)

        response = orchestrator.handle_message(
            user_id=user_id,
            text=text,
            workspace_id=workspace_id,
            conversation_window=conversation_window,
            progress_callback=progress_update,
        )
        if channel_id and response_ts:
            try:
                client.chat_update(channel=channel_id, ts=response_ts, text=response)
                _append_message_to_db(
                    workspace_id=workspace_id,
                    slack_user_id=user_id,
                    channel_id=channel_id,
                    role="assistant",
                    text=response,
                    slack_ts=response_ts,
                )
                return
            except Exception as exc:
                logger.info("Could not update final streaming message: %s", exc)
        sent = say(text=response)
        _append_message_to_db(
            workspace_id=workspace_id,
            slack_user_id=user_id,
            channel_id=channel_id,
            role="assistant",
            text=response,
            slack_ts=str(getattr(sent, "get", lambda *_: "")("ts") or ""),
        )

    @slack_app.event("app_mention")
    def ignore_mentions(body, logger):  # noqa: ANN001
        logger.info("Ignoring app_mention event in DM-only mode")


def _load_dm_conversation_window_from_db(
    workspace_id: str,
    slack_user_id: str,
    channel_id: str,
    limit: int,
) -> str:
    if not channel_id:
        return ""
    try:
        with SessionLocal() as db:
            return load_conversation_window(
                db=db,
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
                slack_channel_id=channel_id,
                limit=limit,
            )
    except Exception as exc:
        logger.info("Could not load DM history from DB: %s", exc)
        return ""


def _persist_dm_context(
    workspace_id: str,
    slack_user_id: str,
    conversation_window: str,
    last_user_message: str,
) -> None:
    try:
        with SessionLocal() as db:
            set_user_context_entry(
                db=db,
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
                context_key="latest_dm_window",
                value={"text": conversation_window},
            )
            set_user_context_entry(
                db=db,
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
                context_key="last_user_message",
                value={"text": last_user_message},
            )
            db.commit()
    except Exception as exc:
        logger.info("Could not persist user DM context: %s", exc)


def _append_message_to_db(
    workspace_id: str,
    slack_user_id: str,
    channel_id: str,
    role: str,
    text: str,
    slack_ts: str = "",
) -> None:
    if not channel_id or not text:
        return
    try:
        with SessionLocal() as db:
            append_conversation_message(
                db=db,
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
                slack_channel_id=channel_id,
                role=role,
                text=text,
                slack_ts=slack_ts or None,
            )
            db.commit()
    except Exception as exc:
        logger.info("Could not append conversation message to DB: %s", exc)
