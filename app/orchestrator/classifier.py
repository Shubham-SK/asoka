from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm.client import get_claude_client

CLASSIFIER_PROMPT = """
You classify Slack DM requests for a Salesforce assistant.

Return ONLY valid JSON:
{"intent":"read_request|write_request|context_edit|approval_response|role_scope_query|plan_management","reason":"short reason"}

Definitions:
- read_request: asks for information; read-only Salesforce operations.
- write_request: asks to create/update/delete Salesforce data or metadata.
- context_edit: coworker is explicitly adding/modifying policy/context for the bot.
- approval_response: coworker is approving/rejecting/requesting changes on a previously created plan.
- role_scope_query: asks about user role, permissions, or what actions they are allowed to perform.
- plan_management: asks to list/show/check/open pending plans or plan queue state.

Rules:
- Use best judgment from semantics, not keyword matching.
- context_edit and approval_response are only valid when is_coworker=true; otherwise choose read_request or write_request.
- Keep reason concise.
"""


@dataclass
class ClassificationResult:
    intent: str
    reason: str


def classify_message(
    text: str,
    is_coworker: bool,
    settings: Settings,
    conversation_window: str = "",
) -> ClassificationResult:
    if not settings.llm_enabled or settings.llm_provider != "anthropic":
        return ClassificationResult(
            intent="read_request",
            reason="llm unavailable, defaulting to read_request",
        )

    try:
        client = get_claude_client()
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=200,
            temperature=0,
            system=CLASSIFIER_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Classify this message:\n"
                        f"recent_conversation_window={conversation_window or '<none>'}\n"
                        f"text={text}\n"
                        f"is_coworker={str(is_coworker).lower()}"
                    ),
                }
            ],
        )

        raw_text = _extract_text(response)
        payload = _extract_json_object(raw_text)
        intent = str(payload.get("intent", "read_request")).strip()
        reason = str(payload.get("reason", "model classification")).strip()
    except Exception as exc:
        return ClassificationResult(
            intent="read_request",
            reason=f"classifier_error_default_read: {type(exc).__name__}",
        )

    allowed_intents = {
        "read_request",
        "write_request",
        "context_edit",
        "approval_response",
        "role_scope_query",
        "plan_management",
    }
    if intent not in allowed_intents:
        intent = "read_request"
        reason = f"invalid_intent_from_model: {payload.get('intent')}"
    if intent == "context_edit" and not is_coworker:
        intent = "read_request"
        reason = "context_edit rejected for non-coworker"
    if intent == "approval_response" and not is_coworker:
        intent = "read_request"
        reason = "approval_response rejected for non-coworker"
    return ClassificationResult(intent=intent, reason=reason)


def _extract_text(response: Any) -> str:
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts).strip()


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()
    return json.loads(raw_text)
