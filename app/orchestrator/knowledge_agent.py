from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.db.enums import (
    ConfidenceTier,
    KnowledgeKind,
    KnowledgeLifecycleStatus,
    KnowledgeQuestionStatus,
)
from app.db.repository import (
    create_knowledge_item,
    delete_knowledge_item,
    get_knowledge_item_by_id,
    list_knowledge_items,
    update_knowledge_item,
)
from app.db.session import SessionLocal
from app.llm.client import get_claude_client

logger = logging.getLogger(__name__)

KNOWLEDGE_AGENT_PROMPT = """
You manage CRUD operations for knowledge instances in a Salesforce assistant.

Return ONLY valid JSON in one of these forms:
{"action":"tool","tool":"list_knowledge","input":{"limit":25,"query":"optional","kinds":["fact|rule|trend|hypothesis|question"],"include_superseded":false},"reason":"..."}
{"action":"tool","tool":"get_knowledge","input":{"knowledge_id":"<uuid>"},"reason":"..."}
{"action":"tool","tool":"create_knowledge","input":{"kind":"fact|rule|trend|hypothesis|question","confidence_tier":"strict_violation|similar_past_approval|observed_trend|coworker_context","title":"...","statement":"...","confidence_score":0.8,"sf_object_api_name":"optional","sf_field_api_name":"optional","question_status":"open|resolved|dismissed"},"reason":"..."}
{"action":"tool","tool":"update_knowledge","input":{"knowledge_id":"<uuid>","title":"optional","statement":"optional","kind":"optional","confidence_tier":"optional","confidence_score":0.7,"sf_object_api_name":"optional","sf_field_api_name":"optional","question_status":"optional","lifecycle_status":"active|superseded"},"reason":"..."}
{"action":"tool","tool":"delete_knowledge","input":{"knowledge_id":"<uuid>"},"reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- For create, include at least title + statement and sensible defaults.
- For update/delete, fetch the target first if ID context is unclear.
- Keep final answers concise for Slack mrkdwn.
- Do not use markdown tables, headings (`#`), markdown links `[text](url)`, or `**bold**`.
- Use Slack mrkdwn style (`*bold*`) and Slack link format (`<https://example.com|label>`).
"""


def run_knowledge_agent(
    settings: Settings,
    user_text: str,
    workspace_id: str,
    requester_slack_user_id: str,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    conversation_window: str = "",
    max_steps: int = 8,
) -> str:
    client = get_claude_client()
    events: list[dict[str, str]] = []
    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Execution context:\n"
                f"workspace_id={workspace_id}\n"
                f"requester_slack_user_id={requester_slack_user_id}\n"
                f"recent_conversation_window={conversation_window or '<none>'}\n"
            ),
        },
        {"role": "user", "content": f"User request:\n{user_text}\n\nDecide first action."},
    ]

    for step in range(1, max_steps + 1):
        action = _next_action(client=client, model=settings.llm_model, transcript=transcript)
        action_type = str(action.get("action", "")).strip()
        if action_type == "final":
            answer = str(action.get("answer", "")).strip() or "Knowledge request completed."
            return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + answer
        if action_type != "tool":
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\nI could not complete this knowledge request."
            )
        tool = str(action.get("tool", "")).strip()
        payload = action.get("input", {})
        reason = str(action.get("reason", "")).strip() or "Run tool."
        events.append({"step": str(step), "type": tool or "tool", "status": "started", "reason": reason})
        try:
            result = _run_tool(tool=tool, payload=payload, workspace_id=workspace_id)
            events[-1]["status"] = "success"
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append({"role": "user", "content": f"Tool result:\n{json.dumps(result, ensure_ascii=True)}"})
        except Exception as exc:
            logger.info("Knowledge agent tool failed: %s", exc)
            events[-1]["status"] = "error"
            events[-1]["reason"] = f"{reason} ({type(exc).__name__}: {exc})"
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append({"role": "user", "content": f"Tool error: {type(exc).__name__}: {exc}"})
    return (
        _build_observability_blob(events, parsed_intent, parsed_intent_reason)
        + "\n\nI reached the step limit while managing knowledge."
    )


def _run_tool(tool: str, payload: dict[str, Any], workspace_id: str) -> dict[str, Any]:
    if tool == "list_knowledge":
        limit = payload.get("limit", 25)
        if not isinstance(limit, int):
            limit = 25
        include_superseded = bool(payload.get("include_superseded", False))
        query = str(payload.get("query", "")).strip() or None
        raw_kinds = payload.get("kinds", [])
        kinds: list[KnowledgeKind] = []
        if isinstance(raw_kinds, list):
            for val in raw_kinds:
                text = str(val).strip()
                for enum_val in KnowledgeKind:
                    if enum_val.value == text:
                        kinds.append(enum_val)
        with SessionLocal() as db:
            items = list_knowledge_items(
                db=db,
                workspace_id=workspace_id,
                limit=limit,
                include_superseded=include_superseded,
                kinds=kinds or None,
                query=query,
            )
            serialized_items = [_serialize_item(item) for item in items]
        return {"items": serialized_items, "count": len(serialized_items)}

    if tool == "get_knowledge":
        knowledge_id = str(payload.get("knowledge_id", "")).strip()
        if not knowledge_id:
            return {"error": "get_knowledge requires `knowledge_id`."}
        with SessionLocal() as db:
            item = get_knowledge_item_by_id(db=db, workspace_id=workspace_id, knowledge_id=knowledge_id)
            if item is None:
                return {"error": f"No knowledge item found for id {knowledge_id}."}
            serialized_item = _serialize_item(item)
        return {"item": serialized_item}

    if tool == "create_knowledge":
        kind = _parse_kind(payload.get("kind"))
        tier = _parse_tier(payload.get("confidence_tier"))
        title = str(payload.get("title", "")).strip()
        statement = str(payload.get("statement", "")).strip()
        if not title or not statement:
            return {"error": "create_knowledge requires `title` and `statement`."}
        content = {"statement": statement}
        question_status = _parse_question_status(payload.get("question_status"))
        with SessionLocal() as db:
            item = create_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                kind=kind,
                confidence_tier=tier,
                title=title,
                content=content,
                confidence_score=_parse_score(payload.get("confidence_score")),
                sf_object_api_name=str(payload.get("sf_object_api_name", "")).strip() or None,
                sf_field_api_name=str(payload.get("sf_field_api_name", "")).strip() or None,
                question_status=question_status,
                provenance={"source": "knowledge_management"},
            )
            db.commit()
            db.refresh(item)
            serialized_item = _serialize_item(item)
        return {"created": serialized_item}

    if tool == "update_knowledge":
        knowledge_id = str(payload.get("knowledge_id", "")).strip()
        if not knowledge_id:
            return {"error": "update_knowledge requires `knowledge_id`."}
        with SessionLocal() as db:
            item = update_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                knowledge_id=knowledge_id,
                title=_opt_str(payload.get("title")),
                statement=_opt_str(payload.get("statement")),
                kind=_opt_kind(payload.get("kind")),
                confidence_tier=_opt_tier(payload.get("confidence_tier")),
                confidence_score=_opt_score(payload.get("confidence_score")),
                sf_object_api_name=_opt_str(payload.get("sf_object_api_name")),
                sf_field_api_name=_opt_str(payload.get("sf_field_api_name")),
                question_status=_opt_question_status(payload.get("question_status")),
                lifecycle_status=_opt_lifecycle(payload.get("lifecycle_status")),
            )
            if item is None:
                return {"error": f"No knowledge item found for id {knowledge_id}."}
            db.commit()
            db.refresh(item)
            serialized_item = _serialize_item(item)
        return {"updated": serialized_item}

    if tool == "delete_knowledge":
        knowledge_id = str(payload.get("knowledge_id", "")).strip()
        if not knowledge_id:
            return {"error": "delete_knowledge requires `knowledge_id`."}
        with SessionLocal() as db:
            ok = delete_knowledge_item(db=db, workspace_id=workspace_id, knowledge_id=knowledge_id)
            db.commit()
        if not ok:
            return {"error": f"No knowledge item found for id {knowledge_id}."}
        return {"deleted": knowledge_id, "lifecycle_status": "superseded"}

    return {"error": f"Unknown tool: {tool}"}


def _serialize_item(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind.value,
        "confidence_tier": item.confidence_tier.value,
        "confidence_score": float(item.confidence_score),
        "title": item.title,
        "statement": str((item.content_json or {}).get("statement", "")).strip(),
        "sf_object_api_name": item.sf_object_api_name,
        "sf_field_api_name": item.sf_field_api_name,
        "lifecycle_status": item.lifecycle_status.value,
        "question_status": item.question_status.value if item.question_status else None,
        "usage_count": int(item.usage_count or 0),
        "updated_at": item.updated_at.isoformat(),
    }


def _parse_kind(value: Any) -> KnowledgeKind:
    text = str(value or "").strip()
    for item in KnowledgeKind:
        if item.value == text:
            return item
    return KnowledgeKind.fact


def _parse_tier(value: Any) -> ConfidenceTier:
    text = str(value or "").strip()
    for item in ConfidenceTier:
        if item.value == text:
            return item
    return ConfidenceTier.coworker_context


def _parse_question_status(value: Any) -> KnowledgeQuestionStatus | None:
    text = str(value or "").strip()
    for item in KnowledgeQuestionStatus:
        if item.value == text:
            return item
    return None


def _parse_lifecycle(value: Any) -> KnowledgeLifecycleStatus:
    text = str(value or "").strip()
    for item in KnowledgeLifecycleStatus:
        if item.value == text:
            return item
    return KnowledgeLifecycleStatus.active


def _parse_score(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except Exception:
        return 0.8


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _opt_kind(value: Any) -> KnowledgeKind | None:
    if value is None:
        return None
    return _parse_kind(value)


def _opt_tier(value: Any) -> ConfidenceTier | None:
    if value is None:
        return None
    return _parse_tier(value)


def _opt_question_status(value: Any) -> KnowledgeQuestionStatus | None:
    if value is None:
        return None
    return _parse_question_status(value)


def _opt_lifecycle(value: Any) -> KnowledgeLifecycleStatus | None:
    if value is None:
        return None
    return _parse_lifecycle(value)


def _opt_score(value: Any) -> float | None:
    if value is None:
        return None
    return _parse_score(value)


def _next_action(client: Any, model: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=700,
        temperature=0,
        system=KNOWLEDGE_AGENT_PROMPT,
        messages=transcript,
    )
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    raw = "\n".join(text_parts).strip()
    try:
        return _extract_json_object(raw)
    except Exception:
        if raw:
            return {"action": "final", "answer": raw}
        return {"action": "final", "answer": "I could not parse the knowledge action."}


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        candidate = _extract_first_json_object_text(raw_text)
        if candidate is not None:
            return json.loads(candidate)
        raise


def _extract_first_json_object_text(raw_text: str) -> str | None:
    start = raw_text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : idx + 1]
    return None


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
        lines.append(f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}")
    return "```\n" + "\n".join(lines) + "\n```"
