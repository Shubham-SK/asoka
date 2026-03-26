from __future__ import annotations

from typing import Any

from app.agent.artifacts import (
    artifact_extract,
    artifact_keys,
    artifact_put,
    artifact_search,
    artifact_tree,
)
from app.salesforce.client import get_salesforce_client


def sf_describe_object(
    object_name: str,
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    sf = get_salesforce_client(slack_user_id=slack_user_id, workspace_id=workspace_id)
    return sf.__getattr__(object_name).describe()


def sf_query_read_only(
    soql: str,
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    normalized = soql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only read-only SELECT SOQL is allowed in Phase 1.")

    sf = get_salesforce_client(slack_user_id=slack_user_id, workspace_id=workspace_id)
    return sf.query(soql)


def sf_tooling_query(
    soql: str,
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Read-only Tooling API query surface (metadata-focused).
    Example: SELECT Id, ValidationName, EntityDefinitionId FROM ValidationRule LIMIT 20
    """
    normalized = soql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only read-only SELECT SOQL is allowed for tooling queries.")

    sf = get_salesforce_client(slack_user_id=slack_user_id, workspace_id=workspace_id)
    return sf.restful("tooling/query", params={"q": soql})


def artifact_store(payload: Any, source: str) -> dict[str, Any]:
    return artifact_put(payload=payload, source=source)


def artifact_list_keys(artifact_id: str, path: str = "") -> dict[str, Any]:
    return artifact_keys(artifact_id=artifact_id, path=path)


def artifact_get_tree(artifact_id: str, path: str = "", max_depth: int = 2) -> dict[str, Any]:
    return artifact_tree(artifact_id=artifact_id, path=path, max_depth=max_depth)


def artifact_search_text(artifact_id: str, query: str, max_hits: int = 20) -> dict[str, Any]:
    return artifact_search(artifact_id=artifact_id, query=query, max_hits=max_hits)


def artifact_extract_path(artifact_id: str, path: str, max_chars: int = 4000) -> dict[str, Any]:
    return artifact_extract(artifact_id=artifact_id, path=path, max_chars=max_chars)
