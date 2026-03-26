from __future__ import annotations

import json
import logging
from typing import Any
from typing import Callable

from app.config import Settings
from app.db.enums import PlanStatus
from app.db.repository import (
    create_execution_plan,
    list_plan_summaries,
    list_pending_plan_summaries,
    set_execution_plan_status,
)
from app.db.session import SessionLocal
from app.llm.client import get_claude_client

logger = logging.getLogger(__name__)

PLAN_AGENT_PROMPT = """
You are a workflow orchestrator for write-plan management in a Salesforce assistant.

Return ONLY valid JSON in one of these forms:
{"action":"tool","tool":"create_plan","input":{"summary":"...","operations":[{"step":1,"type":"...","description":"..."}],"assumptions":[{"type":"...","value":"..."}]},"reason":"..."}
{"action":"tool","tool":"list_plans","input":{"statuses":["pending_approval|approved|denied|draft|executed|failed"],"requester":"self|any|U123","limit":25},"reason":"..."}
{"action":"tool","tool":"list_pending_plans","input":{"limit":25},"reason":"..."}
{"action":"tool","tool":"approve_plan","input":{"plan_id":"<uuid>","reason":"..."},"reason":"..."}
{"action":"tool","tool":"reject_plan","input":{"plan_id":"<uuid>","reason":"..."},"reason":"..."}
{"action":"tool","tool":"role_scope","input":{},"reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- Use tools to inspect pending plans before approving/rejecting if the plan_id is unclear.
- For non-coworkers, never approve/reject plans; explain limitation in final answer.
- For non-coworkers, only query plans for requester=self.
- Keep final answers concise for Slack mrkdwn.
- Do not use markdown tables; use bullet lists or short paragraphs.
- After create_plan, include exact token `Plan ID: \`<uuid>\`` in final answer.
- If create_plan result status is pending_approval, include exact token `Status: \`pending_approval\``.
- If approve/reject tools return `needs_clarification=true`, ask the user to pick one plan ID explicitly.
- For status questions like "denied?", "approved?", or "all plans", use `list_plans` (not `list_pending_plans`).
"""


def run_plan_agent(
    settings: Settings,
    user_text: str,
    workspace_id: str,
    requester_slack_user_id: str,
    is_coworker: bool,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    conversation_window: str = "",
    max_steps: int = 10,
    notify_pending_plan_callback: Callable[[str, str, str, str], None] | None = None,
    notify_plan_status_callback: Callable[[str, str, str, str, str, str], None] | None = None,
) -> str:
    client = get_claude_client()
    events: list[dict[str, str]] = []
    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Execution context:\n"
                f"is_coworker={str(is_coworker).lower()}\n"
                f"workspace_id={workspace_id}\n"
                f"requester_slack_user_id={requester_slack_user_id}\n"
                f"recent_conversation_window={conversation_window or '<none>'}\n"
            ),
        },
        {
            "role": "user",
            "content": f"User request:\n{user_text}\n\nDecide first action.",
        },
    ]

    for _ in range(max_steps):
        action = _next_action(client=client, model=settings.llm_model, transcript=transcript)
        action_type = str(action.get("action", "")).strip()
        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            if answer:
                return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + answer
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\n"
                + "I could not produce a final plan response."
            )

        if action_type != "tool":
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\n"
                + "I could not process that plan workflow request."
            )

        tool = str(action.get("tool", "")).strip()
        payload = action.get("input", {})
        action_reason = str(action.get("reason", "")).strip() or "no reason provided"
        if not isinstance(payload, dict):
            payload = {}

        events.append(
            {
                "step": str(len(events) + 1),
                "type": "tool",
                "status": "started",
                "reason": action_reason,
                "input": f"{tool}({json.dumps(payload, ensure_ascii=True)})",
                "output": "",
            }
        )
        try:
            result = _run_tool(
                tool=tool,
                payload=payload,
                workspace_id=workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                is_coworker=is_coworker,
                user_text=user_text,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
                notify_pending_plan_callback=notify_pending_plan_callback,
                notify_plan_status_callback=notify_plan_status_callback,
            )
            events[-1]["status"] = "success"
            events[-1]["output"] = _truncate_text(json.dumps(result, ensure_ascii=True), 220)
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool}): {json.dumps(result, ensure_ascii=True)}",
                }
            )
        except Exception as exc:
            logger.info("Plan agent tool failed: %s", exc)
            events[-1]["status"] = "error"
            events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append(
                {
                    "role": "user",
                    "content": f"Tool error ({tool}): {type(exc).__name__}: {exc}",
                }
            )

    return (
        _build_observability_blob(events, parsed_intent, parsed_intent_reason)
        + "\n\n"
        + "I reached the plan orchestration step limit. Please retry with a more specific request."
    )


def _run_tool(
    tool: str,
    payload: dict[str, Any],
    workspace_id: str,
    requester_slack_user_id: str,
    is_coworker: bool,
    user_text: str,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    notify_pending_plan_callback: Callable[[str, str, str, str], None] | None = None,
    notify_plan_status_callback: Callable[[str, str, str, str, str, str], None] | None = None,
) -> dict[str, Any]:
    if tool == "role_scope":
        if is_coworker:
            return {
                "role": "coworker",
                "allowed": ["read_request", "write_request", "approve_or_reject_plans"],
            }
        return {
            "role": "end_user",
            "allowed": ["read_request", "write_request_plan_creation_only"],
        }

    if tool == "create_plan":
        summary = str(payload.get("summary", "")).strip() or user_text.strip()[:500]
        operations = payload.get("operations")
        assumptions = payload.get("assumptions")
        if not isinstance(operations, list) or not operations:
            operations = [
                {
                    "step": 1,
                    "type": "pending_write_design",
                    "description": "Mutating Salesforce operations will be finalized before execution.",
                }
            ]
        if not isinstance(assumptions, list):
            assumptions = []
        assumptions.append({"type": "intent", "value": "write_request"})
        if parsed_intent:
            assumptions.append({"type": "intent_parse", "value": parsed_intent})
        if parsed_intent_reason:
            assumptions.append({"type": "intent_reason", "value": parsed_intent_reason})

        with SessionLocal() as db:
            status = PlanStatus.approved if is_coworker else PlanStatus.pending_approval
            plan = create_execution_plan(
                db=db,
                workspace_id=workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                summary=summary,
                operations=operations,
                assumptions=assumptions,
                safety_checks=[],
                status=status,
            )
            plan_id = plan.id
            db.commit()
        if (
            status == PlanStatus.pending_approval
            and not is_coworker
            and notify_pending_plan_callback is not None
        ):
            try:
                notify_pending_plan_callback(
                    plan_id,
                    workspace_id,
                    requester_slack_user_id,
                    summary,
                )
            except Exception as exc:
                logger.info("Could not notify coworker for plan=%s: %s", plan_id, exc)
        return {"plan_id": plan_id, "status": status.value}

    if tool == "list_pending_plans":
        limit = payload.get("limit", 25)
        if not isinstance(limit, int):
            limit = 25
        limit = max(1, min(limit, 50))
        requester = None if is_coworker else requester_slack_user_id
        with SessionLocal() as db:
            plans = list_pending_plan_summaries(
                db=db,
                workspace_id=workspace_id,
                requester_slack_user_id=requester,
                limit=limit,
            )
        return {"pending_plans": plans}

    if tool == "list_plans":
        limit = payload.get("limit", 25)
        if not isinstance(limit, int):
            limit = 25
        limit = max(1, min(limit, 50))

        requester_raw = str(payload.get("requester", "self")).strip().lower()
        requester: str | None
        if not is_coworker:
            requester = requester_slack_user_id
        elif requester_raw in {"self", ""}:
            requester = requester_slack_user_id
        elif requester_raw == "any":
            requester = None
        else:
            requester = str(payload.get("requester", "")).strip() or requester_slack_user_id

        statuses_raw = payload.get("statuses", [])
        statuses: list[PlanStatus] = []
        if isinstance(statuses_raw, list):
            for value in statuses_raw:
                text = str(value).strip().lower()
                for enum_val in PlanStatus:
                    if enum_val.value == text:
                        statuses.append(enum_val)
                        break

        with SessionLocal() as db:
            plans = list_plan_summaries(
                db=db,
                workspace_id=workspace_id,
                statuses=statuses or None,
                requester_slack_user_id=requester,
                limit=limit,
            )
        return {
            "plans": plans,
            "filters": {
                "statuses": [s.value for s in statuses],
                "requester": requester or "any",
                "limit": limit,
            },
        }

    if tool == "approve_plan":
        if not is_coworker:
            return {"error": "Only coworker can approve plans."}
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            with SessionLocal() as db:
                plans = list_pending_plan_summaries(db=db, workspace_id=workspace_id, limit=10)
            return {
                "needs_clarification": True,
                "message": "Please specify which plan to approve.",
                "pending_plans": plans,
            }
        reason = str(payload.get("reason", "")).strip()
        with SessionLocal() as db:
            try:
                plan = set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.approved,
                    reason=reason,
                    actor_slack_user_id=requester_slack_user_id,
                    allowed_from_statuses=[PlanStatus.pending_approval],
                )
            except ValueError as exc:
                return {"error": str(exc)}
            if plan is None:
                return {"error": f"No plan found: {plan_id}"}
            target_requester = plan.requester_slack_user_id
            db.commit()
        if notify_plan_status_callback is not None:
            try:
                notify_plan_status_callback(
                    plan_id,
                    workspace_id,
                    target_requester,
                    PlanStatus.approved.value,
                    reason,
                    requester_slack_user_id,
                )
            except Exception as exc:
                logger.info("Could not notify requester for approved plan=%s: %s", plan_id, exc)
        return {"plan_id": plan_id, "status": PlanStatus.approved.value}

    if tool == "reject_plan":
        if not is_coworker:
            return {"error": "Only coworker can reject plans."}
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            with SessionLocal() as db:
                plans = list_pending_plan_summaries(db=db, workspace_id=workspace_id, limit=10)
            return {
                "needs_clarification": True,
                "message": "Please specify which plan to reject.",
                "pending_plans": plans,
            }
        reason = str(payload.get("reason", "")).strip()
        with SessionLocal() as db:
            try:
                plan = set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.denied,
                    reason=reason,
                    actor_slack_user_id=requester_slack_user_id,
                    allowed_from_statuses=[PlanStatus.pending_approval],
                )
            except ValueError as exc:
                return {"error": str(exc)}
            if plan is None:
                return {"error": f"No plan found: {plan_id}"}
            target_requester = plan.requester_slack_user_id
            db.commit()
        if notify_plan_status_callback is not None:
            try:
                notify_plan_status_callback(
                    plan_id,
                    workspace_id,
                    target_requester,
                    PlanStatus.denied.value,
                    reason,
                    requester_slack_user_id,
                )
            except Exception as exc:
                logger.info("Could not notify requester for denied plan=%s: %s", plan_id, exc)
        return {"plan_id": plan_id, "status": PlanStatus.denied.value, "reason": reason}

    return {"error": f"Unknown tool: {tool}"}


def _next_action(client: Any, model: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=PLAN_AGENT_PROMPT,
        messages=transcript,
    )
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    raw = "\n".join(text_parts).strip()
    return _extract_json_object(raw)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()
    return json.loads(raw_text)


def _build_observability_blob(
    events: list[dict[str, str]],
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> str:
    lines = ["Execution trace"]
    if parsed_intent:
        lines.append(f"- Intent parse: {parsed_intent}")
    if parsed_intent_reason:
        lines.append(f"- Intent reason: {parsed_intent_reason}")
    if not events:
        lines.append("- No tool calls were executed.")
        return "```\n" + "\n".join(lines) + "\n```"
    for event in events:
        lines.append(
            f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."
