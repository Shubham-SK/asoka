from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.agent.tools import (
    artifact_extract_path,
    artifact_get_tree,
    artifact_list_keys,
    artifact_search_text,
    artifact_store,
    sf_describe_object,
    sf_query_read_only,
    sf_tooling_query,
)
from app.config import Settings
from app.llm.client import get_claude_client

logger = logging.getLogger(__name__)

ARTIFACT_MIN_BYTES = 6000
ARTIFACT_MIN_RECORDS = 20

SYSTEM_PROMPT = """
You are a Salesforce read-only assistant.
You may only use read operations.

At each step, return ONLY valid JSON with one of these shapes:
{"action":"query","soql":"SELECT ...","reason":"..."}
{"action":"tooling_query","soql":"SELECT ...","reason":"..."}
{"action":"describe","object":"Account","reason":"..."}
{"action":"artifact_list_keys","artifact_id":"art_xxx","path":"fields","reason":"..."}
{"action":"artifact_get_tree","artifact_id":"art_xxx","path":"fields[0]","max_depth":2,"reason":"..."}
{"action":"artifact_search_text","artifact_id":"art_xxx","query":"Type","max_hits":20,"reason":"..."}
{"action":"artifact_extract_path","artifact_id":"art_xxx","path":"fields[0].name","max_chars":4000,"reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- Never propose or execute write operations.
- Keep SOQL safe and read-only (SELECT only).
- If a query result is enough, return action=final with concise answer.
- For action=final, format answer for Slack mrkdwn, not GitHub markdown.
- Do not use markdown tables; use bullet lists or fenced code blocks instead.
- Do not use GitHub/CommonMark constructs like headings (`#`), markdown links `[text](url)`, or `**bold**`.
- For emphasis, use Slack mrkdwn (`*bold*`), and for links use Slack format (`<https://example.com|label>`).
- Keep final answers brief and scannable for DM.
- If a tool call fails, inspect the error and try a corrected query/action.
- Use tooling_query when you need metadata/config entities such as ValidationRule.
- Large JSON outputs should be explored via artifact_* tools instead of asking for full dumps.
- After query/describe/tooling_query success, use returned artifact_id for follow-up extraction.
- Do not guess object names, field names, stage names, picklist values, record types, asset names, or other identifiers.
- When any requested name is uncertain, do discovery first by listing/describing metadata and then use the exact discovered names.
- Prefer discovery-first plans: identify valid objects/fields/values before constructing filtered queries.
- If user phrasing is ambiguous, resolve it by discovery queries rather than assumptions.
- For org-specific "current state" questions, execute at least one relevant tool call before returning final.
"""

FORCED_FINAL_PROMPT = """
You are producing a best-effort final response for Slack.
Use the prior transcript and tool outputs.

Requirements:
- Output plain Slack mrkdwn text only (no JSON).
- Do not use GitHub/CommonMark constructs like headings (`#`), markdown links `[text](url)`, or `**bold**`.
- For emphasis, use Slack mrkdwn (`*bold*`), and for links use Slack format (`<https://example.com|label>`).
- Summarize what is known with high confidence.
- Clearly call out uncertainties and failed tool attempts.
- Keep concise and actionable.
"""


def run_read_agent(
    settings: Settings,
    user_text: str,
    slack_user_id: str = "",
    workspace_id: str = "",
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    conversation_window: str = "",
    max_steps: int = 25,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    client = get_claude_client()
    attempts: list[str] = []
    events: list[dict[str, Any]] = []
    context_block = (
        "Recent DM conversation window (oldest to newest):\n"
        f"{_truncate_text(conversation_window, 12000) if conversation_window else '<none>'}\n\n"
        "Use this only to resolve references like 'these accounts'. "
        "If older context conflicts with the current request, prefer the current request."
    )
    intent_block = (
        "Classifier context (authoritative routing hint):\n"
        f"- intent: {parsed_intent or '<none>'}\n"
        f"- reason: {_truncate_text(parsed_intent_reason, 800) if parsed_intent_reason else '<none>'}\n\n"
        "Keep tool selection aligned with this intent/reason unless the user explicitly changes topic."
    )
    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": intent_block,
        },
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
    enforced_tool_call_retry_used = False

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
                _emit_progress_update(
                    events, progress_callback, parsed_intent, parsed_intent_reason
                )
                return (
                    _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                    + "\n\n"
                    + forced
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            return _build_failure_summary(
                user_text=user_text,
                attempts=attempts,
                reason="I could not parse a valid next action from the model.",
            )
        action_type = action.get("action")
        action_reason = _truncate_text(str(action.get("reason", "")).strip(), 220)

        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            if (
                not events
                and not enforced_tool_call_retry_used
                and _looks_like_org_state_question(user_text)
            ):
                enforced_tool_call_retry_used = True
                transcript.append({"role": "assistant", "content": json.dumps(action)})
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned final without any tool calls. "
                            "This request asks for current org state. "
                            "Execute at least one relevant tool call first."
                        ),
                    }
                )
                continue
            if answer:
                return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + answer
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
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            try:
                result = sf_describe_object(
                    object_name,
                    slack_user_id=slack_user_id or None,
                    workspace_id=workspace_id or None,
                )
                summary = _summarize_describe_result(result)
                materialized = _materialize_result_for_model(
                    result=result,
                    source=f"describe:{object_name}",
                    include_rows_preview=False,
                )
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = _materialized_output_summary(summary, materialized)
                tool_content = f"Tool result (describe {object_name}):\n{materialized['model_payload']}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
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
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
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
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            try:
                result = sf_query_read_only(
                    soql,
                    slack_user_id=slack_user_id or None,
                    workspace_id=workspace_id or None,
                )
                summary = _summarize_query_result(result)
                materialized = _materialize_result_for_model(
                    result=result,
                    source="query",
                    include_rows_preview=True,
                )
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = _materialized_output_summary(summary, materialized)
                events[-1]["rows_preview"] = materialized.get("rows_preview", [])
                tool_content = f"Tool result (query):\n{materialized['model_payload']}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
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
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
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
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            try:
                result = sf_tooling_query(
                    soql,
                    slack_user_id=slack_user_id or None,
                    workspace_id=workspace_id or None,
                )
                summary = _summarize_query_result(result)
                materialized = _materialize_result_for_model(
                    result=result,
                    source="tooling_query",
                    include_rows_preview=True,
                )
                attempts.append(f"  -> success: {summary}")
                events[-1]["status"] = "success"
                events[-1]["output"] = _materialized_output_summary(summary, materialized)
                events[-1]["rows_preview"] = materialized.get("rows_preview", [])
                tool_content = f"Tool result (tooling_query):\n{materialized['model_payload']}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
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
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        if action_type == "artifact_list_keys":
            artifact_id = str(action.get("artifact_id", "")).strip()
            path = str(action.get("path", "")).strip()
            attempts.append(
                f"step {step + 1}: artifact_list_keys artifact_id={artifact_id or '<missing>'} path={path or '<root>'}"
            )
            events.append(
                {
                    "step": str(step + 1),
                    "type": "artifact_list_keys",
                    "reason": action_reason or "no reason provided",
                    "input": f"artifact_id={artifact_id}, path={path or '<root>'}",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                result = artifact_list_keys(artifact_id=artifact_id, path=path)
                events[-1]["status"] = "success"
                events[-1]["output"] = _truncate_text(json.dumps(result, ensure_ascii=True), 220)
                tool_content = f"Tool result (artifact_list_keys):\n{json.dumps(result, ensure_ascii=True)}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (artifact_list_keys): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        if action_type == "artifact_get_tree":
            artifact_id = str(action.get("artifact_id", "")).strip()
            path = str(action.get("path", "")).strip()
            max_depth = int(action.get("max_depth", 2) or 2)
            attempts.append(
                f"step {step + 1}: artifact_get_tree artifact_id={artifact_id or '<missing>'} path={path or '<root>'}"
            )
            events.append(
                {
                    "step": str(step + 1),
                    "type": "artifact_get_tree",
                    "reason": action_reason or "no reason provided",
                    "input": f"artifact_id={artifact_id}, path={path or '<root>'}, max_depth={max_depth}",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                result = artifact_get_tree(artifact_id=artifact_id, path=path, max_depth=max_depth)
                events[-1]["status"] = "success"
                events[-1]["output"] = "returned tree summary"
                tool_content = f"Tool result (artifact_get_tree):\n{_truncate_json(result, 5000)}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (artifact_get_tree): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        if action_type == "artifact_search_text":
            artifact_id = str(action.get("artifact_id", "")).strip()
            query = str(action.get("query", "")).strip()
            max_hits = int(action.get("max_hits", 20) or 20)
            attempts.append(
                f"step {step + 1}: artifact_search_text artifact_id={artifact_id or '<missing>'} query={_truncate_text(query, 60)}"
            )
            events.append(
                {
                    "step": str(step + 1),
                    "type": "artifact_search_text",
                    "reason": action_reason or "no reason provided",
                    "input": f"artifact_id={artifact_id}, query={query}, max_hits={max_hits}",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                result = artifact_search_text(
                    artifact_id=artifact_id, query=query, max_hits=max_hits
                )
                events[-1]["status"] = "success"
                events[-1]["output"] = f"hit_count={result.get('hit_count', 0)}"
                tool_content = f"Tool result (artifact_search_text):\n{_truncate_json(result, 5000)}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (artifact_search_text): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        if action_type == "artifact_extract_path":
            artifact_id = str(action.get("artifact_id", "")).strip()
            path = str(action.get("path", "")).strip()
            max_chars = int(action.get("max_chars", 4000) or 4000)
            attempts.append(
                f"step {step + 1}: artifact_extract_path artifact_id={artifact_id or '<missing>'} path={_truncate_text(path, 80)}"
            )
            events.append(
                {
                    "step": str(step + 1),
                    "type": "artifact_extract_path",
                    "reason": action_reason or "no reason provided",
                    "input": f"artifact_id={artifact_id}, path={path}, max_chars={max_chars}",
                    "status": "started",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                result = artifact_extract_path(
                    artifact_id=artifact_id, path=path, max_chars=max_chars
                )
                events[-1]["status"] = "success"
                events[-1]["output"] = f"value_type={result.get('value_type')}"
                tool_content = f"Tool result (artifact_extract_path):\n{_truncate_json(result, 5000)}"
                events[-1]["model_preview"] = _truncate_text(tool_content, 320)
                transcript.append({"role": "user", "content": tool_content})
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (artifact_extract_path): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
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
        _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
        return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + forced
    _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
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
        candidate = _extract_first_json_object_text(raw_text)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Invalid JSON from LLM: {raw_text}") from exc


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


def _build_observability_blob(
    events: list[dict[str, Any]],
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


def _extract_rows_preview(result: dict[str, Any], max_rows: int = 10) -> list[str]:
    records = result.get("records")
    if not isinstance(records, list):
        return []

    preview: list[str] = []
    for raw_record in records[:max_rows]:
        cleaned = _strip_salesforce_attributes(raw_record)
        dumped = json.dumps(cleaned, ensure_ascii=True)
        preview.append(_truncate_text(dumped, 450))
    return preview


def _strip_salesforce_attributes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_salesforce_attributes(v)
            for k, v in value.items()
            if k != "attributes"
        }
    if isinstance(value, list):
        return [_strip_salesforce_attributes(item) for item in value]
    return value


def _materialize_result_for_model(result: Any, source: str, include_rows_preview: bool) -> dict[str, Any]:
    size_bytes = _json_size_bytes(result)
    rows_preview = _extract_rows_preview(result) if include_rows_preview else []
    record_count = _record_count(result)
    should_artifact = size_bytes >= ARTIFACT_MIN_BYTES or record_count >= ARTIFACT_MIN_RECORDS

    if should_artifact:
        artifact = artifact_store(result, source=source)
        payload = {
            "mode": "artifact",
            "artifact": artifact,
            "rows_preview": rows_preview,
            "size_bytes": size_bytes,
        }
        return {
            "mode": "artifact",
            "artifact_id": artifact["artifact_id"],
            "rows_preview": rows_preview,
            "model_payload": json.dumps(payload, ensure_ascii=True),
            "size_bytes": size_bytes,
        }

    payload = {
        "mode": "inline",
        "size_bytes": size_bytes,
        "result": result,
    }
    return {
        "mode": "inline",
        "artifact_id": None,
        "rows_preview": rows_preview,
        "model_payload": _truncate_json(payload, 5000),
        "size_bytes": size_bytes,
    }


def _materialized_output_summary(summary: str, materialized: dict[str, Any]) -> str:
    if materialized["mode"] == "artifact":
        return f"{summary} (artifact_id={materialized['artifact_id']}, size={materialized['size_bytes']} bytes)"
    return f"{summary} (inline, size={materialized['size_bytes']} bytes)"


def _json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True).encode("utf-8"))


def _record_count(value: Any) -> int:
    if isinstance(value, dict):
        records = value.get("records")
        if isinstance(records, list):
            return len(records)
    return 0


def _looks_like_org_state_question(user_text: str) -> bool:
    text = user_text.strip().lower()
    cues = (
        "any ",
        "current",
        "as it stands",
        "how many",
        "what are",
        "show",
        "list",
        "settings",
        "validation",
        "enforcement",
        "rule",
        "minimum",
        "maximum",
        "field",
        "opportunity",
        "account",
        "case",
        "pipeline",
        "deals",
    )
    return any(cue in text for cue in cues)


def _emit_progress_update(
    events: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> None:
    if progress_callback is None or not events:
        return
    try:
        progress_callback(
            _build_observability_blob(events, parsed_intent, parsed_intent_reason)
            + "\n\n_Working..._"
        )
    except Exception as exc:
        logger.info("Progress callback failed: %s", exc)
