from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.tools import sf_describe_object, sf_query_read_only, sf_tooling_query
from app.config import Settings
from app.llm.client import get_claude_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a Salesforce read-only assistant.
You may only use read operations.

At each step, return ONLY valid JSON with one of these shapes:
{"action":"query","soql":"SELECT ...","reason":"..."}
{"action":"tooling_query","soql":"SELECT ...","reason":"..."}
{"action":"describe","object":"Account","reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- Never propose or execute write operations.
- Keep SOQL safe and read-only (SELECT only).
- If a query result is enough, return action=final with concise answer.
- For action=final, format answer for Slack mrkdwn, not GitHub markdown.
- Do not use markdown tables; use bullet lists or fenced code blocks instead.
- Keep final answers brief and scannable for DM.
- If a tool call fails, inspect the error and try a corrected query/action.
- Use tooling_query when you need metadata/config entities such as ValidationRule.
"""

FORCED_FINAL_PROMPT = """
You are producing a best-effort final response for Slack.
Use the prior transcript and tool outputs.

Requirements:
- Output plain Slack mrkdwn text only (no JSON).
- Summarize what is known with high confidence.
- Clearly call out uncertainties and failed tool attempts.
- Keep concise and actionable.
"""


def run_read_agent(
    settings: Settings,
    user_text: str,
    conversation_window: str = "",
    max_steps: int = 25,
) -> str:
    client = get_claude_client()
    attempts: list[str] = []
    events: list[dict[str, str]] = []
    context_block = (
        "Recent DM conversation window (oldest to newest):\n"
        f"{_truncate_text(conversation_window, 12000) if conversation_window else '<none>'}\n\n"
        "Use this to resolve references like 'these accounts'."
    )
    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": context_block,
        },
        {
            "role": "user",
            "content": (
                "User request:\n"
                f"{user_text}\n\n"
                "Decide first action."
            ),
        }
    ]

    for step in range(max_steps):
        try:
            action = _next_action(client=client, model=settings.llm_model, transcript=transcript)
        except Exception as exc:
            attempts.append(f"step {step + 1}: model output parsing failed ({type(exc).__name__})")
            forced = _force_finalize(
                client=client,
                model=settings.llm_model,
                user_text=user_text,
                transcript=transcript,
                attempts=attempts,
                reason="I could not parse a valid next action from the model.",
            )
            if forced:
                return _build_observability_blob(events) + "\n\n" + forced
            return _build_failure_summary(
                user_text=user_text,
                attempts=attempts,
                reason="I could not parse a valid next action from the model.",
            )
        action_type = action.get("action")
        action_reason = _truncate_text(str(action.get("reason", "")).strip(), 220)

        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            if answer:
                return _build_observability_blob(events) + "\n\n" + answer
            attempts.append(f"step {step + 1}: model returned empty final answer")
            return _build_failure_summary(
                user_text=user_text,
                attempts=attempts,
                reason="The model returned an empty final answer.",
            )

        if action_type == "describe":
            object_name = str(action.get("object", "")).strip()
            attempts.append(f"step {step + 1}: describe `{object_name or '<missing>'}`")
            events.append(
                {
                    "step": str(step + 1),
                    "type": "describe",
                    "reason": action_reason or "no reason provided",
                    "input": object_name or "<missing>",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action),
                }
            )
            if not object_name:
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool error (describe): missing object name. "
                            "Provide a valid Salesforce object API name."
                        ),
                    }
                )
                attempts.append(f"  -> error: missing object name")
                events[-1]["status"] = "error"
                events[-1]["output"] = "missing object name"
                continue
            try:
                result = sf_describe_object(object_name)
                summary = _summarize_describe_result(result)
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = summary
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool result (describe {object_name}):\n{_truncate_json(result)}",
                    }
                )
            except Exception as exc:
                logger.info("Describe tool failed, allowing model retry: %s", exc)
                attempts.append(f"  -> error: {type(exc).__name__}")
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (describe {object_name}): {type(exc).__name__}: {exc}",
                    }
                )
            continue

        if action_type == "query":
            soql = str(action.get("soql", "")).strip()
            attempts.append(f"step {step + 1}: query `{_truncate_text(soql, 140) or '<missing>'}`")
            events.append(
                {
                    "step": str(step + 1),
                    "type": "query",
                    "reason": action_reason or "no reason provided",
                    "input": _truncate_text(soql, 220) or "<missing>",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action),
                }
            )
            if not soql:
                transcript.append(
                    {
                        "role": "user",
                        "content": "Tool error (query): missing SOQL. Provide a valid SELECT query.",
                    }
                )
                attempts.append("  -> error: missing SOQL")
                events[-1]["status"] = "error"
                events[-1]["output"] = "missing SOQL"
                continue
            try:
                result = sf_query_read_only(soql)
                summary = _summarize_query_result(result)
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = summary
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool result (query):\n{_truncate_json(result)}",
                    }
                )
            except Exception as exc:
                logger.info("Query tool failed, allowing model retry: %s", exc)
                attempts.append(f"  -> error: {type(exc).__name__}")
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (query): {type(exc).__name__}: {exc}",
                    }
                )
            continue

        if action_type == "tooling_query":
            soql = str(action.get("soql", "")).strip()
            attempts.append(
                f"step {step + 1}: tooling_query `{_truncate_text(soql, 140) or '<missing>'}`"
            )
            events.append(
                {
                    "step": str(step + 1),
                    "type": "tooling_query",
                    "reason": action_reason or "no reason provided",
                    "input": _truncate_text(soql, 220) or "<missing>",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action),
                }
            )
            if not soql:
                transcript.append(
                    {
                        "role": "user",
                        "content": "Tool error (tooling_query): missing SOQL. Provide a valid SELECT query.",
                    }
                )
                attempts.append("  -> error: missing SOQL")
                events[-1]["status"] = "error"
                events[-1]["output"] = "missing SOQL"
                continue
            try:
                result = sf_tooling_query(soql)
                summary = _summarize_query_result(result)
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = summary
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool result (tooling_query):\n{_truncate_json(result)}",
                    }
                )
            except Exception as exc:
                logger.info("Tooling query failed, allowing model retry: %s", exc)
                attempts.append(f"  -> error: {type(exc).__name__}")
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (tooling_query): {type(exc).__name__}: {exc}",
                    }
                )
            continue

        logger.warning("Unexpected agent action: %s", action)
        attempts.append(f"step {step + 1}: unexpected action `{action_type}`")
        return _build_failure_summary(
            user_text=user_text,
            attempts=attempts,
            reason="The model produced an unsupported action type.",
        )

    limit_reason = f"I reached the read-only step limit ({max_steps}) before a final answer."
    forced = _force_finalize(
        client=client,
        model=settings.llm_model,
        user_text=user_text,
        transcript=transcript,
        attempts=attempts,
        reason=limit_reason,
    )
    if forced:
        return _build_observability_blob(events) + "\n\n" + forced
    return _build_failure_summary(
        user_text=user_text,
        attempts=attempts,
        reason=limit_reason,
    )


def _next_action(client: Any, model: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=transcript,
    )
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    raw = "\n".join(text_parts).strip()
    return _extract_json_object(raw)


def _force_finalize(
    client: Any,
    model: str,
    user_text: str,
    transcript: list[dict[str, str]],
    attempts: list[str],
    reason: str,
) -> str:
    """
    Force a best-effort final answer when the action loop fails or reaches limits.
    """
    summary_tail = "\n".join(f"- {item}" for item in attempts[-12:])
    final_messages = list(transcript) + [
        {
            "role": "user",
            "content": (
                "The tool loop must stop now.\n"
                f"Original request: {user_text}\n"
                f"Stop reason: {reason}\n"
                "Recent attempt summary:\n"
                f"{summary_tail if summary_tail else '- no attempts recorded'}\n\n"
                "Return a best-effort final answer now."
            ),
        }
    ]
    try:
        response = client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0,
            system=FORCED_FINAL_PROMPT,
            messages=final_messages,
        )
        text_parts: list[str] = []
        for part in response.content:
            if getattr(part, "type", "") == "text":
                text_parts.append(part.text)
        text = "\n".join(text_parts).strip()
        return text
    except Exception as exc:
        logger.info("Forced finalization failed: %s", exc)
        return ""


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from LLM: {raw_text}") from exc


def _truncate_json(value: Any, max_chars: int = 8000) -> str:
    dumped = json.dumps(value, ensure_ascii=True)
    if len(dumped) <= max_chars:
        return dumped
    return dumped[: max_chars - 50] + '..."[truncated]"'


def _summarize_query_result(result: dict[str, Any]) -> str:
    total = result.get("totalSize")
    if isinstance(total, int):
        return f"{total} row(s)"
    if "records" in result and isinstance(result["records"], list):
        return f"{len(result['records'])} record(s)"
    return "query returned data"


def _summarize_describe_result(result: dict[str, Any]) -> str:
    object_name = result.get("name")
    fields = result.get("fields")
    if isinstance(fields, list):
        return f"{object_name or 'object'} has {len(fields)} field(s)"
    return "describe returned metadata"


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _build_failure_summary(user_text: str, attempts: list[str], reason: str) -> str:
    lines = [
        "*I could not complete that request yet.*",
        f"- Reason: {reason}",
        f"- Request: {_truncate_text(user_text, 180)}",
    ]
    if attempts:
        lines.append("- What I tried:")
        for item in attempts[-8:]:
            lines.append(f"  - {item}")
    lines.append("- Next step: rephrase the request or ask for a narrower query scope.")
    return "\n".join(lines)


def _build_observability_blob(events: list[dict[str, str]]) -> str:
    lines = ["*Execution trace*"]
    if not events:
        lines.append("- No tool calls were executed.")
        return "\n".join(lines)

    for event in events:
        lines.append(
            f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}"
        )
        lines.append(f"  - Input: {event['input']}")
        lines.append(f"  - Output: {event['output']}")
    return "\n".join(lines)
