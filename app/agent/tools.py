from __future__ import annotations

from typing import Any

from app.salesforce.client import get_salesforce_client


def sf_describe_object(object_name: str) -> dict[str, Any]:
    sf = get_salesforce_client()
    return sf.__getattr__(object_name).describe()


def sf_query_read_only(soql: str) -> dict[str, Any]:
    normalized = soql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only read-only SELECT SOQL is allowed in Phase 1.")

    sf = get_salesforce_client()
    return sf.query(soql)


def sf_tooling_query(soql: str) -> dict[str, Any]:
    """
    Read-only Tooling API query surface (metadata-focused).
    Example: SELECT Id, ValidationName, EntityDefinitionId FROM ValidationRule LIMIT 20
    """
    normalized = soql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only read-only SELECT SOQL is allowed for tooling queries.")

    sf = get_salesforce_client()
    return sf.restful("tooling/query", params={"q": soql})
