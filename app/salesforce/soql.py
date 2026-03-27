from __future__ import annotations

from typing import Any

from app.salesforce.client import get_salesforce_client


def ensure_read_only_select(soql: str, context: str = "SOQL") -> str:
    query = (soql or "").strip()
    normalized = query.lower()
    if not normalized.startswith("select"):
        raise ValueError(f"Only read-only SELECT {context} is allowed.")
    return query


def escape_soql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def execute_read_query(
    soql: str,
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    query = ensure_read_only_select(soql, context="SOQL")
    sf = get_salesforce_client(slack_user_id=slack_user_id, workspace_id=workspace_id)
    return sf.query(query)


def execute_tooling_query(
    soql: str,
    slack_user_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    query = ensure_read_only_select(soql, context="tooling query")
    sf = get_salesforce_client(slack_user_id=slack_user_id, workspace_id=workspace_id)
    return sf.restful("tooling/query", params={"q": query})
