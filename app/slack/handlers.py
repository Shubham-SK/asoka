from __future__ import annotations

import json
import logging

from app.config import Settings
from app.db.enums import PlanStatus
from app.db.repository import (
    append_conversation_message,
    get_execution_plan_for_workspace,
    load_conversation_window,
    set_execution_plan_status,
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
                    text="Thinking...",
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

        def plan_notification_callback(
            plan_id: str,
            callback_workspace_id: str,
            requester_slack_user_id: str,
            summary: str,
        ) -> None:
            _notify_coworker_pending_plan(
                client=client,
                settings=settings,
                workspace_id=callback_workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                plan_id=plan_id,
                summary=summary,
            )

        def plan_status_notification_callback(
            plan_id: str,
            callback_workspace_id: str,
            requester_slack_user_id: str,
            status: str,
            reason: str,
            actor_slack_user_id: str,
        ) -> None:
            _notify_requester_plan_status(
                client=client,
                plan_id=plan_id,
                workspace_id=callback_workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                status=status,
                reason=reason,
                actor_slack_user_id=actor_slack_user_id,
            )

        response = orchestrator.handle_message(
            user_id=user_id,
            text=text,
            workspace_id=workspace_id,
            conversation_window=conversation_window,
            progress_callback=progress_update,
            plan_notification_callback=plan_notification_callback,
            plan_status_notification_callback=plan_status_notification_callback,
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

    @slack_app.action("approve_plan_button")
    def handle_approve_plan_button(ack, body, client, logger):  # noqa: ANN001
        ack()
        actor_user_id = str(body.get("user", {}).get("id", "") or "")
        if actor_user_id != settings.slack_coworker_user_id:
            if actor_user_id:
                client.chat_postMessage(
                    channel=actor_user_id,
                    text="Only the designated human coworker can approve plans.",
                )
            return

        action_list = body.get("actions", [])
        if not action_list:
            return
        raw_value = str(action_list[0].get("value", "") or "")
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            client.chat_postMessage(channel=actor_user_id, text="Could not parse approve action payload.")
            return

        workspace_id = str(payload.get("workspace_id", "") or "default")
        plan_id = str(payload.get("plan_id", "") or "")
        requester_slack_user_id = str(payload.get("requester_slack_user_id", "") or "")
        if not plan_id:
            client.chat_postMessage(channel=actor_user_id, text="Approve action missing plan_id.")
            return

        with SessionLocal() as db:
            try:
                plan = set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.approved,
                    reason="approved via Slack button",
                    actor_slack_user_id=actor_user_id,
                    allowed_from_statuses=[PlanStatus.pending_approval],
                )
            except ValueError as exc:
                client.chat_postMessage(
                    channel=actor_user_id,
                    text=f"Cannot approve plan `{plan_id}`: {exc}",
                )
                return
            if plan is None:
                client.chat_postMessage(
                    channel=actor_user_id,
                    text=f"No plan found for ID `{plan_id}` in workspace `{workspace_id}`.",
                )
                return
            requester_slack_user_id = requester_slack_user_id or plan.requester_slack_user_id
            db.commit()

        client.chat_postMessage(
            channel=actor_user_id,
            text=f"Plan `{plan_id}` approved.",
        )
        if requester_slack_user_id:
            _notify_requester_plan_status(
                client=client,
                plan_id=plan_id,
                workspace_id=workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                status=PlanStatus.approved.value,
                reason="approved via Slack button",
                actor_slack_user_id=actor_user_id,
            )


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


def _notify_coworker_pending_plan(
    client,
    settings: Settings,
    workspace_id: str,
    requester_slack_user_id: str,
    plan_id: str,
    summary: str,
) -> None:
    if not settings.plan_notify_coworker_on_create:
        return
    if requester_slack_user_id == settings.slack_coworker_user_id:
        return
    if not settings.slack_coworker_user_id:
        return

    msg = (
        "New write plan pending approval.\n"
        f"- Workspace: `{workspace_id}`\n"
        f"- Requester: <@{requester_slack_user_id}>\n"
        f"- Plan ID: `{plan_id}`\n"
        "You can approve via button, or reply with `approve/reject plan <plan_id> ...`."
    )
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*New write plan pending approval*\n"
                    f"Workspace: `{workspace_id}`\n"
                    f"Requester: <@{requester_slack_user_id}>\n"
                    f"Plan ID: `{plan_id}`\n"
                    f"Summary: {summary}"
                ),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve Plan"},
                    "style": "primary",
                    "action_id": "approve_plan_button",
                    "value": json.dumps(
                        {
                            "workspace_id": workspace_id,
                            "plan_id": plan_id,
                            "requester_slack_user_id": requester_slack_user_id,
                        },
                        ensure_ascii=True,
                    ),
                }
            ],
        },
    ]
    try:
        client.chat_postMessage(
            channel=settings.slack_coworker_user_id,
            text=msg,
            blocks=blocks,
        )
    except Exception as exc:
        logger.info("Could not notify coworker about pending plan: %s", exc)


def _notify_requester_plan_status(
    client,
    plan_id: str,
    workspace_id: str,
    requester_slack_user_id: str,
    status: str,
    reason: str,
    actor_slack_user_id: str,
) -> None:
    if not requester_slack_user_id:
        return
    status_label = str(status).strip().lower()
    if status_label == PlanStatus.approved.value:
        text = (
            f"Your write plan `{plan_id}` was *approved* by <@{actor_slack_user_id}>.\n"
            f"Workspace: `{workspace_id}`\n"
            "Execution can proceed once executor wiring is enabled."
        )
    elif status_label == PlanStatus.denied.value:
        reason_text = reason.strip() or "No reason provided."
        text = (
            f"Your write plan `{plan_id}` was *denied* by <@{actor_slack_user_id}>.\n"
            f"Workspace: `{workspace_id}`\n"
            f"Feedback: {reason_text}"
        )
    else:
        reason_text = reason.strip() or "No additional feedback."
        text = (
            f"Your write plan `{plan_id}` status changed to `{status_label}` by <@{actor_slack_user_id}>.\n"
            f"Workspace: `{workspace_id}`\n"
            f"Feedback: {reason_text}"
        )
    try:
        client.chat_postMessage(channel=requester_slack_user_id, text=text)
    except Exception as exc:
        logger.info("Could not notify requester about plan status update: %s", exc)
