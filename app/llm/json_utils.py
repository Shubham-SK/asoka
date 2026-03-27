from __future__ import annotations

import json
from typing import Any


def extract_text_response(response: Any) -> str:
    text_parts: list[str] = []
    for part in getattr(response, "content", []):
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts).strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        candidate = _extract_first_json_object_text(text)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON from LLM: {text}") from exc
        else:
            raise RuntimeError(f"Invalid JSON from LLM: {text}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Expected top-level JSON object from LLM.")
    return parsed


def extract_json_from_response(response: Any) -> dict[str, Any]:
    return extract_json_object(extract_text_response(response))


def repair_json_object_with_llm(
    client: Any,
    model: str,
    raw_text: str,
    schema_hint: str,
    max_tokens: int = 1000,
) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    system_prompt = (
        "You are a strict JSON repair tool. Convert the provided malformed output into ONE valid JSON object.\n"
        f"Expected schema/context: {schema_hint}\n"
        "Return JSON only, no markdown, no prose."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": text[:7000]}],
        )
        return extract_json_from_response(response)
    except Exception:
        return None


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
