from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.llm.client import get_claude_client
from app.llm.json_utils import extract_json_from_response

CLASSIFIER_PROMPT = """
You classify Slack DM requests for a Salesforce assistant.

Return ONLY valid JSON:
{"intent":"read_request|write_request|approval_response|role_scope_query|plan_management|knowledge_ingestion|knowledge_management","reason":"short reason"}

Definitions:
- read_request: asks for information; read-only Salesforce operations.
- write_request: asks to create/update/delete Salesforce data or metadata.
- approval_response: coworker is approving/rejecting/requesting changes on a previously created plan.
- role_scope_query: asks about user role, permissions, or what actions they are allowed to perform.
- plan_management: asks to list/show/check/open pending plans or plan queue state.
- knowledge_ingestion: coworker explicitly asks to ingest/update structured knowledge from recent outputs/context.
- knowledge_management: coworker asks to query/create/update/delete knowledge instances, or to add/modify policy/context guidance for future behavior.

Rules:
- Use best judgment from semantics, not keyword matching.
- approval_response, knowledge_ingestion, and knowledge_management are only valid when is_coworker=true; otherwise choose read_request or write_request.
- knowledge_ingestion must be explicit. Only choose it when user clearly asks to ingest/extract/persist/rebuild knowledge.
- If user is answering a prior assistant question, giving context, or discussing plans/next steps without explicitly requesting ingestion, do NOT choose knowledge_ingestion.
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

        payload = extract_json_from_response(response)
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
        "approval_response",
        "role_scope_query",
        "plan_management",
        "knowledge_ingestion",
        "knowledge_management",
    }
    if intent not in allowed_intents:
        intent = "read_request"
        reason = f"invalid_intent_from_model: {payload.get('intent')}"
    if intent == "context_edit":
        intent = "knowledge_management"
        reason = "context_edit remapped to knowledge_management"
    if intent == "approval_response" and not is_coworker:
        intent = "read_request"
        reason = "approval_response rejected for non-coworker"
    if intent == "knowledge_ingestion" and not is_coworker:
        intent = "read_request"
        reason = "knowledge_ingestion rejected for non-coworker"
    if intent == "knowledge_management" and not is_coworker:
        intent = "read_request"
        reason = "knowledge_management rejected for non-coworker"
    return ClassificationResult(intent=intent, reason=reason)
