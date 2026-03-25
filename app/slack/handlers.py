from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.orchestrator.service import Orchestrator

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

        if not text or not user_id:
            logger.info("Skipping empty DM event")
            return

        conversation_window = _load_dm_conversation_window(
            client=client,
            channel_id=event.get("channel", ""),
            current_event_ts=event.get("ts", ""),
            limit=25,
        )
        logger.info("Received DM from user=%s text=%s", user_id, text)
        response = orchestrator.handle_message(
            user_id=user_id,
            text=text,
            conversation_window=conversation_window,
        )
        say(text=response)

    @slack_app.event("app_mention")
    def ignore_mentions(body, logger):  # noqa: ANN001
        logger.info("Ignoring app_mention event in DM-only mode")


def _load_dm_conversation_window(
    client: Any,
    channel_id: str,
    current_event_ts: str,
    limit: int,
) -> str:
    if not channel_id:
        return ""
    try:
        result = client.conversations_history(channel=channel_id, limit=limit)
        messages = result.get("messages", [])
    except Exception as exc:
        logger.info("Could not load DM history: %s", exc)
        return ""

    ordered = list(reversed(messages))
    lines: list[str] = []
    for msg in ordered:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if current_event_ts and msg.get("ts") == current_event_ts:
            continue

        role = "assistant" if msg.get("bot_id") or msg.get("subtype") == "bot_message" else "user"
        lines.append(f"{role}: {text}")

    return "\n".join(lines)
