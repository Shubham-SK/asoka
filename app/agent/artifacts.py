from __future__ import annotations

import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class Artifact:
    artifact_id: str
    source: str
    payload: Any


class ArtifactStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: dict[str, Artifact] = {}

    def put(self, payload: Any, source: str) -> dict[str, Any]:
        artifact_id = f"art_{uuid4().hex[:12]}"
        artifact = Artifact(artifact_id=artifact_id, source=source, payload=payload)
        with self._lock:
            self._items[artifact_id] = artifact

        return {
            "artifact_id": artifact_id,
            "source": source,
            "summary": _payload_summary(payload),
            "top_level_keys": _top_keys(payload),
        }

    def get(self, artifact_id: str) -> Artifact:
        with self._lock:
            if artifact_id not in self._items:
                raise KeyError(f"Unknown artifact_id: {artifact_id}")
            return self._items[artifact_id]


STORE = ArtifactStore()


def artifact_put(payload: Any, source: str) -> dict[str, Any]:
    return STORE.put(payload=payload, source=source)


def artifact_keys(artifact_id: str, path: str = "") -> dict[str, Any]:
    value = _resolve_path(STORE.get(artifact_id).payload, path)
    if isinstance(value, dict):
        return {"path": path or "<root>", "keys": list(value.keys())}
    if isinstance(value, list):
        return {"path": path or "<root>", "keys": [f"[{i}]" for i in range(len(value))]}
    return {"path": path or "<root>", "keys": [], "note": "target is scalar"}


def artifact_tree(artifact_id: str, path: str = "", max_depth: int = 2) -> dict[str, Any]:
    value = _resolve_path(STORE.get(artifact_id).payload, path)
    return {
        "path": path or "<root>",
        "tree": _tree_summary(value, depth=max_depth),
    }


def artifact_search(artifact_id: str, query: str, max_hits: int = 20) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query is required")
    payload = STORE.get(artifact_id).payload
    hits: list[dict[str, str]] = []
    _search(payload, query.lower(), path="", hits=hits, max_hits=max_hits)
    return {"query": query, "hit_count": len(hits), "hits": hits}


def artifact_extract(
    artifact_id: str,
    path: str,
    max_chars: int = 4000,
) -> dict[str, Any]:
    value = _resolve_path(STORE.get(artifact_id).payload, path)
    rendered = json.dumps(value, ensure_ascii=True)
    truncated = len(rendered) > max_chars
    if truncated:
        rendered = rendered[: max_chars - 17] + '..."[truncated]"'
    return {
        "path": path,
        "value_type": type(value).__name__,
        "rendered": rendered,
        "truncated": truncated,
    }


def _payload_summary(payload: Any) -> str:
    if isinstance(payload, dict):
        return f"dict with {len(payload)} key(s)"
    if isinstance(payload, list):
        return f"list with {len(payload)} item(s)"
    return f"scalar {type(payload).__name__}"


def _top_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        return list(payload.keys())[:100]
    return []


def _resolve_path(payload: Any, path: str) -> Any:
    if not path or path == "/":
        return payload
    cursor = payload
    for token in _parse_path(path):
        if isinstance(token, int):
            if not isinstance(cursor, list):
                raise KeyError(f"Expected list for index access: [{token}]")
            cursor = cursor[token]
        else:
            if not isinstance(cursor, dict):
                raise KeyError(f"Expected object for key access: {token}")
            cursor = cursor[token]
    return cursor


def _parse_path(path: str) -> list[str | int]:
    """
    Dot+index path parser:
    fields[0].name
    fields.0.name
    """
    cleaned = path.strip().lstrip(".")
    tokens: list[str | int] = []
    for part in cleaned.split("."):
        if not part:
            continue
        m = re.fullmatch(r"([^\[\]]+)(\[\d+\])*", part)
        if not m:
            raise KeyError(f"Invalid path segment: {part}")
        key = m.group(1)
        tokens.append(key)
        index_parts = re.findall(r"\[(\d+)\]", part)
        for idx in index_parts:
            tokens.append(int(idx))
        if part.isdigit():
            tokens.pop()
            tokens.append(int(part))
    return tokens


def _tree_summary(value: Any, depth: int) -> Any:
    if depth <= 0:
        return _payload_summary(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k in list(value.keys())[:25]:
            out[k] = _tree_summary(value[k], depth - 1)
        if len(value) > 25:
            out["..."] = f"{len(value) - 25} more key(s)"
        return out
    if isinstance(value, list):
        sample = [_tree_summary(v, depth - 1) for v in value[:5]]
        if len(value) > 5:
            sample.append(f"... {len(value) - 5} more item(s)")
        return sample
    return value


def _search(value: Any, query_lower: str, path: str, hits: list[dict[str, str]], max_hits: int) -> None:
    if len(hits) >= max_hits:
        return

    if isinstance(value, dict):
        for key, val in value.items():
            key_path = f"{path}.{key}" if path else key
            if query_lower in key.lower():
                hits.append({"path": key_path, "match": "key"})
                if len(hits) >= max_hits:
                    return
            _search(val, query_lower, key_path, hits, max_hits)
            if len(hits) >= max_hits:
                return
        return

    if isinstance(value, list):
        for i, item in enumerate(value):
            list_path = f"{path}[{i}]"
            _search(item, query_lower, list_path, hits, max_hits)
            if len(hits) >= max_hits:
                return
        return

    scalar = str(value)
    if query_lower in scalar.lower():
        hits.append({"path": path or "<root>", "match": "value"})
