from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.config import Settings
from app.llm.client import get_claude_client
from app.mcp.salesforce_client import SalesforceMcpClient, SalesforceMcpTool

logger = logging.getLogger(__name__)

MCP_SYSTEM_PROMPT = """
You are a Salesforce read-only assistant operating through MCP tools.

At each step, return ONLY valid JSON in one of these forms:
{"action":"list_tools","reason":"..."}
{"action":"call_tool","tool":"tool_name","input":{"key":"value"},"reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- Read-only only. Never perform write/mutating operations.
- Prefer discovery first: inspect available tools before calling unfamiliar ones.
- If an MCP tool fails, inspect the error and try a corrected read-only call.
- Keep final answer concise for Slack mrkdwn.
- Do not use markdown tables, headings (`#`), markdown links `[text](url)`, or `**bold**`.
- For emphasis, use Slack mrkdwn (`*bold*`), and for links use Slack format (`<https://example.com|label>`).
"""


def run_mcp_read_agent(
    settings: Settings,
    user_text: str,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    conversation_window: str = "",
    max_steps: int = 20,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    client = get_claude_client()
    mcp_client = SalesforceMcpClient(settings=settings)
    events: list[dict[str, str]] = []
    safe_tools_cache: list[SalesforceMcpTool] = []

    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Classifier context:\n"
                f"- intent: {parsed_intent or '<none>'}\n"
                f"- reason: {_truncate_text(parsed_intent_reason, 800) if parsed_intent_reason else '<none>'}\n\n"
                "Recent conversation window:\n"
                f"{_truncate_text(conversation_window, 8000) if conversation_window else '<none>'}\n\n"
                "User request:\n"
                f"{user_text}\n\n"
                "Decide first action."
            ),
        }
    ]

    for step in range(max_steps):
        action = _next_action(client=client, model=settings.llm_model, transcript=transcript)
        action_type = str(action.get("action", "")).strip()
        action_reason = str(action.get("reason", "")).strip() or "no reason provided"

        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            if answer:
                return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + answer
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\nI could not produce a final MCP answer."
            )

        if action_type == "list_tools":
            events.append(
                {
                    "step": str(step + 1),
                    "type": "list_tools",
                    "status": "started",
                    "reason": action_reason,
                    "input": "{}",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                tools = mcp_client.list_tools()
                safe_tools_cache = [tool for tool in tools if _is_non_mutating_tool_name(tool.name)]
                result = {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in safe_tools_cache
                    ]
                }
                events[-1]["status"] = "success"
                events[-1]["output"] = f"safe_tools={len(safe_tools_cache)}"
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool result (list_tools): {_truncate_json(result, 7000)}",
                    }
                )
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error (list_tools): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        if action_type == "call_tool":
            tool_name = str(action.get("tool", "")).strip()
            tool_input = action.get("input", {})
            if not isinstance(tool_input, dict):
                tool_input = {}
            events.append(
                {
                    "step": str(step + 1),
                    "type": "call_tool",
                    "status": "started",
                    "reason": action_reason,
                    "input": f"{tool_name}({json.dumps(tool_input, ensure_ascii=True)})",
                    "output": "",
                }
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            if not tool_name:
                events[-1]["status"] = "error"
                events[-1]["output"] = "missing tool name"
                transcript.append(
                    {
                        "role": "user",
                        "content": "Tool error (call_tool): missing tool name.",
                    }
                )
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            if not _is_non_mutating_tool_name(tool_name):
                events[-1]["status"] = "error"
                events[-1]["output"] = f"blocked mutating tool: {tool_name}"
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool blocked (call_tool): `{tool_name}` looks mutating. "
                            "Use a read-only tool."
                        ),
                    }
                )
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            if safe_tools_cache and tool_name not in {t.name for t in safe_tools_cache}:
                events[-1]["status"] = "error"
                events[-1]["output"] = "tool not in filtered list"
                transcript.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool error (call_tool): `{tool_name}` is not in the read-only filtered tool list. "
                            "Call list_tools again and pick a listed tool."
                        ),
                    }
                )
                _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
                continue
            try:
                result = mcp_client.call_tool(tool_name=tool_name, arguments=tool_input)
                events[-1]["status"] = "success"
                events[-1]["output"] = _truncate_text(json.dumps(result, ensure_ascii=True), 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool result ({tool_name}): {_truncate_json(result, 10000)}",
                    }
                )
            except Exception as exc:
                events[-1]["status"] = "error"
                events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
                transcript.append(
                    {
                        "role": "user",
                        "content": f"Tool error ({tool_name}): {type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress_update(events, progress_callback, parsed_intent, parsed_intent_reason)
            continue

        events.append(
            {
                "step": str(step + 1),
                "type": "unknown_action",
                "status": "error",
                "reason": action_reason,
                "input": json.dumps(action, ensure_ascii=True),
                "output": "unsupported action",
            }
        )
        return (
            _build_observability_blob(events, parsed_intent, parsed_intent_reason)
            + "\n\nI could not process that MCP action."
        )

    return (
        _build_observability_blob(events, parsed_intent, parsed_intent_reason)
        + "\n\nI reached the MCP step limit. Please try a narrower request."
    )


def _next_action(client: Any, model: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=MCP_SYSTEM_PROMPT,
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


def _is_non_mutating_tool_name(tool_name: str) -> bool:
    normalized = tool_name.strip().lower()
    if not normalized:
        return False
    deny_tokens = (
        "create",
        "update",
        "delete",
        "deploy",
        "assign",
        "write",
        "modify",
        "upsert",
        "insert",
        "promote",
        "commit",
        "checkout",
        "resolve",
        "open_org",
        "mutation",
    )
    if any(token in normalized for token in deny_tokens):
        return False
    allow_tokens = (
        "list",
        "get",
        "describe",
        "query",
        "read",
        "fetch",
        "search",
        "scan",
        "guide",
        "reference",
        "explore",
        "username",
        "test",
    )
    return any(token in normalized for token in allow_tokens)


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


def _emit_progress_update(
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(_build_observability_blob(events, parsed_intent, parsed_intent_reason))
    except Exception as exc:
        logger.info("MCP progress callback failed: %s", exc)


def _truncate_json(value: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=True)
    return _truncate_text(text, max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."

