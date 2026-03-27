from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable

from app.agent.tools import artifact_search_text, artifact_store, sf_describe_object, sf_query_read_only, sf_tooling_query
from app.config import Settings
from app.db.enums import ConfidenceTier, KnowledgeKind, KnowledgeQuestionStatus
from app.db.repository import (
    create_knowledge_item,
    delete_knowledge_item,
    list_knowledge_items,
    resolve_or_supersede_by_canonical_key,
    update_knowledge_item,
)
from app.db.session import SessionLocal
from app.llm.client import get_claude_client
from app.llm.json_utils import extract_json_object
from app.llm.json_utils import extract_text_response
from app.llm.json_utils import repair_json_object_with_llm
from app.salesforce.soql import ensure_read_only_select

logger = logging.getLogger(__name__)
INGESTION_LOG_PREVIEW_CHARS = 12000
DISCOVERY_ARTIFACT_THRESHOLD_CHARS = 5000
DISCOVERY_REPAIR_MAX_ATTEMPTS = 6


@dataclass
class IngestionRunResult:
    status: str
    message: str
    observability: str


@dataclass
class DiscoveryToolSpec:
    key: str
    tool_name: str
    reason: str
    api: str
    query: str = ""
    object_name: str = ""


def _object_key_slug(object_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in object_name)


def _soql_literal(value: str) -> str:
    inner = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{inner}'"


INGESTION_PROMPT = """
You are the Salesforce knowledge ingestion agent.
You have a compact toolset:
- execute_soql (read/tooling)
- describe_object
- grep_artifact
- kb_list, kb_create, kb_update, kb_delete

Run metadata-first ingestion and keep persisting useful knowledge incrementally.
Primary extraction objectives:
1) Describe all objects and fields (including field labels, type, required/writeable, defaults, precision/scale, picklist options).
2) Capture validation rules and hard constraints.
3) For every discovered object, audit Tooling metadata beyond describe (FieldDefinition policy signals and per-object ValidationRule rows) for restrictions, required fields, deprecated fields, and rule inventory.
4) Capture naming conventions and custom schema intent.
5) Capture strictly typed semantics (picklists/enums and field-level limitations).
6) Capture automation and behavior cues (tests, triggers, flows).
7) Capture natural-language descriptions, warnings, limitations, and usage guidance tied to objects/fields/inputs.

Output strict JSON with keys facts, hypotheses, questions.
Prioritize completeness over brevity and avoid arbitrary caps.
"""


def ingest_read_response_into_kb(
    settings: Settings,
    workspace_id: str,
    user_text: str,
    parsed_intent: str,
    parsed_intent_reason: str,
    slack_user_id: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> IngestionRunResult:
    events: list[dict[str, str]] = []
    if not settings.knowledge_ingestion_enabled:
        events.append(
            {
                "step": "1",
                "type": "ingestion_config",
                "status": "skipped",
                "reason": "Knowledge ingestion is disabled by config.",
            }
        )
        return IngestionRunResult(
            status="skipped",
            message="Knowledge ingestion is disabled.",
            observability=_build_observability_blob(
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            ),
        )
    if not settings.llm_enabled or settings.llm_provider != "anthropic":
        events.append(
            {
                "step": "1",
                "type": "ingestion_llm_check",
                "status": "error",
                "reason": "Claude is not configured for ingestion.",
            }
        )
        return IngestionRunResult(
            status="error",
            message="Knowledge ingestion failed: Claude is not configured.",
            observability=_build_observability_blob(
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            ),
        )
    client = get_claude_client()
    discovery_contexts, streamed_inserted, streamed_preview = _run_knowledge_discovery(
        client=client,
        settings=settings,
        workspace_id=workspace_id,
        slack_user_id=slack_user_id,
        events=events,
        progress_callback=progress_callback,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    try:
        events.append(
            {
                "step": _next_step(events),
                "type": "finalize_ingestion",
                "status": "started",
                "reason": "Finalize ingestion and build user digest.",
            }
        )
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        inserted = streamed_inserted
        persisted_preview = streamed_preview
        events[-1]["status"] = "success"
        events[-1]["reason"] = f"Persisted {inserted} knowledge items from incremental tool loop."
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        obs = _build_observability_blob(
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
            execution_plan=_ingestion_execution_plan(),
        )
        digest = _build_persisted_items_digest(persisted_preview)
        return IngestionRunResult(
            status="success",
            message=(
                f"Knowledge ingestion completed. Persisted {inserted} items.\n\n"
                f"{obs}\n\n"
                f"{digest}"
            ),
            observability=obs,
        )
    except Exception as exc:
        events[-1]["status"] = "error"
        events[-1]["reason"] = f"Persistence failed: {type(exc).__name__}: {exc}"
        logger.info("Knowledge ingestion persistence failed: %s", exc)
        return IngestionRunResult(
            status="error",
            message=f"Knowledge ingestion persistence failed: {type(exc).__name__}",
            observability=_build_observability_blob(
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
                execution_plan=_ingestion_execution_plan(),
            ),
        )


def _persist_ingestion_document(
    workspace_id: str,
    document: dict[str, Any],
    max_items: int,
) -> tuple[int, dict[str, list[dict[str, str]]]]:
    facts = document.get("facts", [])
    hypotheses = document.get("hypotheses", [])
    questions = document.get("questions", [])
    inserted = 0
    persisted_preview: dict[str, list[dict[str, str]]] = {
        "facts": [],
        "hypotheses": [],
        "questions": [],
    }
    with SessionLocal() as db:
        for item in facts:
            if not isinstance(item, dict):
                continue
            kind = _parse_kind(str(item.get("kind", "fact")).strip().lower())
            tier = _parse_tier(str(item.get("confidence_tier", "observed_trend")).strip().lower())
            score = _parse_score(item.get("confidence_score", 0.5))
            title = str(item.get("title", "")).strip() or str(item.get("statement", "")).strip()[:120]
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            create_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                kind=kind,
                confidence_tier=tier,
                confidence_score=score,
                title=title or "Extracted fact",
                content={"statement": statement, "source_type": "read_ingestion"},
                provenance={"pipeline": "read_ingestion"},
                sf_object_api_name=_normalize_optional_text(item.get("sf_object_api_name")),
            )
            inserted += 1
            persisted_preview["facts"].append(
                {
                    "title": title or "Extracted fact",
                    "statement": statement,
                    "kind": kind.value,
                }
            )

        for item in questions:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            title = str(item.get("title", "")).strip() or question[:120]
            create_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                kind=KnowledgeKind.question,
                confidence_tier=ConfidenceTier.coworker_context,
                confidence_score=0.8,
                title=title or "Question",
                content={
                    "question": question,
                    "why_needed": str(item.get("why_needed", "")).strip(),
                    "blocking_policy": bool(item.get("blocking_policy", False)),
                    "source_type": "read_ingestion",
                },
                provenance={"pipeline": "read_ingestion"},
                question_status=KnowledgeQuestionStatus.open,
            )
            inserted += 1
            persisted_preview["questions"].append(
                {
                    "title": title or "Question",
                    "question": question,
                    "why_needed": str(item.get("why_needed", "")).strip(),
                }
            )

        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            tier = _parse_tier(str(item.get("confidence_tier", "similar_past_approval")).strip().lower())
            score = _parse_score(item.get("confidence_score", 0.4))
            title = str(item.get("title", "")).strip() or str(item.get("statement", "")).strip()[:120]
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            create_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                kind=KnowledgeKind.hypothesis,
                confidence_tier=tier,
                confidence_score=score,
                title=title or "Hypothesis",
                content={"statement": statement, "source_type": "read_ingestion"},
                provenance={"pipeline": "read_ingestion"},
            )
            inserted += 1
            persisted_preview["hypotheses"].append(
                {
                    "title": title or "Hypothesis",
                    "statement": statement,
                }
            )
        db.commit()
    return inserted, persisted_preview


def _validation_rule_knowledge_parts(item: dict[str, Any]) -> tuple[str, str, str | None, str]:
    name = str(item.get("ValidationName", "")).strip() or "UnnamedValidationRule"
    entity = item.get("EntityDefinition")
    object_name = ""
    if isinstance(entity, dict):
        object_name = str(entity.get("QualifiedApiName", "")).strip()
    display_field = str(item.get("ErrorDisplayField", "")).strip()
    message = str(item.get("ErrorMessage", "")).strip()
    formula = str(item.get("ErrorConditionFormula", "")).strip()
    active = bool(item.get("Active", False))
    description = str(item.get("Description", "")).strip()
    parts = [
        f"Validation rule `{name}` on `{object_name or 'unknown object'}`",
        f"active={active}",
    ]
    if display_field:
        parts.append(f"display_field={display_field}")
    if message:
        parts.append(f"error_message={message}")
    if formula:
        parts.append(f"formula={formula}")
    if description:
        parts.append(f"description={description}")
    statement = "; ".join(parts)
    title = f"Validation Rule: {object_name + '.' if object_name else ''}{name}"
    canonical_key = f"validation_rule:{object_name or 'unknown'}:{name}"
    return title, statement, _normalize_optional_text(object_name), canonical_key


def _field_description_knowledge_parts(
    object_name: str,
    field: dict[str, Any],
) -> tuple[str, str, str]:
    api_name = str(field.get("name", "")).strip()
    if not api_name:
        return "", "", ""
    label = str(field.get("label", "")).strip()
    field_type = str(field.get("type", "")).strip()
    required = (not bool(field.get("nillable", True))) and (not bool(field.get("defaultedOnCreate", False)))
    custom = bool(field.get("custom", False))
    createable = bool(field.get("createable", False))
    updateable = bool(field.get("updateable", False))
    picklist_values = field.get("picklistValues", [])
    picks: list[str] = []
    if isinstance(picklist_values, list):
        for val in picklist_values:
            if not isinstance(val, dict):
                continue
            value = str(val.get("value", "")).strip()
            if value:
                picks.append(value)
    precision = field.get("precision")
    scale = field.get("scale")
    parts = [
        f"Field `{object_name}.{api_name}`",
        f"label={label or '<none>'}",
        f"type={field_type or '<unknown>'}",
        f"required={required}",
        f"custom={custom}",
        f"createable={createable}",
        f"updateable={updateable}",
    ]
    if picks:
        parts.append(f"picklist_values={','.join(picks)}")
    if precision is not None:
        parts.append(f"precision={precision}")
    if scale is not None:
        parts.append(f"scale={scale}")
    statement = "; ".join(parts)
    title = f"Field Description: {object_name}.{api_name}"
    canonical_key = f"field_description:{object_name}:{api_name}"
    return title, statement, canonical_key


def _group_validation_rule_names_by_object(result: Any) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        return grouped
    for item in records:
        if not isinstance(item, dict):
            continue
        entity = item.get("EntityDefinition")
        object_name = ""
        if isinstance(entity, dict):
            object_name = str(entity.get("QualifiedApiName", "")).strip()
        rule_name = str(item.get("ValidationName", "")).strip()
        if not object_name or not rule_name:
            continue
        grouped.setdefault(object_name, [])
        if rule_name not in grouped[object_name]:
            grouped[object_name].append(rule_name)
    return grouped


def _object_policy_knowledge_parts(
    object_name: str,
    describe: dict[str, Any],
    validation_rule_names: list[str],
) -> tuple[str, str, str]:
    fields = describe.get("fields", []) if isinstance(describe, dict) else []
    required_fields: list[str] = []
    non_createable: list[str] = []
    non_updateable: list[str] = []
    restricted_picklists: list[str] = []
    calculated_fields: list[str] = []
    unique_fields: list[str] = []
    if isinstance(fields, list):
        for field in fields:
            if not isinstance(field, dict):
                continue
            api_name = str(field.get("name", "")).strip()
            if not api_name:
                continue
            required = (not bool(field.get("nillable", True))) and (not bool(field.get("defaultedOnCreate", False)))
            if required:
                required_fields.append(api_name)
            if not bool(field.get("createable", False)):
                non_createable.append(api_name)
            if not bool(field.get("updateable", False)):
                non_updateable.append(api_name)
            if bool(field.get("restrictedPicklist", False)):
                restricted_picklists.append(api_name)
            if bool(field.get("calculated", False)):
                calculated_fields.append(api_name)
            if bool(field.get("unique", False)) or bool(field.get("externalId", False)):
                unique_fields.append(api_name)
    parts = [
        f"Object `{object_name}` policy audit",
        f"queryable={bool(describe.get('queryable', False))}",
        f"createable={bool(describe.get('createable', False))}",
        f"updateable={bool(describe.get('updateable', False))}",
        f"deletable={bool(describe.get('deletable', False))}",
        f"required_field_count={len(required_fields)}",
        f"restricted_picklist_count={len(restricted_picklists)}",
        f"formula_field_count={len(calculated_fields)}",
        f"validation_rule_count={len(validation_rule_names)}",
    ]
    if required_fields:
        parts.append(f"required_fields={','.join(required_fields[:30])}")
    if restricted_picklists:
        parts.append(f"restricted_picklists={','.join(restricted_picklists[:30])}")
    if validation_rule_names:
        parts.append(f"validation_rules={','.join(validation_rule_names[:30])}")
    if unique_fields:
        parts.append(f"unique_or_external_id_fields={','.join(unique_fields[:30])}")
    if non_createable:
        parts.append(f"non_createable_field_count={len(non_createable)}")
    if non_updateable:
        parts.append(f"non_updateable_field_count={len(non_updateable)}")
    title = f"Object Policy Audit: {object_name}"
    statement = "; ".join(parts)
    canonical_key = f"object_policy_audit:{object_name}"
    return title, statement, canonical_key


def _summarize_field_definition_policy_counts(records: list[Any]) -> tuple[int, int, int, int, int]:
    total = 0
    required = 0
    restricted_picklist = 0
    calculated = 0
    deprecated = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        total += 1
        if item.get("IsNillable") is False:
            required += 1
        if item.get("IsRestrictedPicklist") is True:
            restricted_picklist += 1
        if item.get("IsCalculated") is True:
            calculated += 1
        if item.get("IsDeprecatedAndHidden") is True:
            deprecated += 1
    return total, required, restricted_picklist, calculated, deprecated


def _summarize_validation_rule_inventory(records: list[Any]) -> tuple[int, int, int]:
    total = 0
    active = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        total += 1
        if item.get("Active") is True:
            active += 1
    inactive = total - active
    return total, active, inactive


def _parse_kind(raw: str) -> KnowledgeKind:
    mapping = {
        "fact": KnowledgeKind.fact,
        "rule": KnowledgeKind.rule,
        "trend": KnowledgeKind.trend,
    }
    return mapping.get(raw, KnowledgeKind.fact)


def _parse_tier(raw: str) -> ConfidenceTier:
    for tier in ConfidenceTier:
        if tier.value == raw:
            return tier
    return ConfidenceTier.coworker_context


def _parse_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(score, 1.0))


def _build_lossless_json_context(label: str, result: Any) -> str:
    try:
        rendered = json.dumps(result, ensure_ascii=True)
    except Exception:
        rendered = str(result)
    return f"{label}:\n{rendered}"


def _run_ingestion_tool(
    *,
    tool: str,
    payload: dict[str, Any],
    workspace_id: str,
    slack_user_id: str,
) -> Any:
    if tool == "execute_soql":
        api = str(payload.get("api", "read")).strip().lower()
        query = str(payload.get("query", "")).strip()
        if api == "tooling":
            return sf_tooling_query(
                query,
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            )
        return sf_query_read_only(
            query,
            slack_user_id=slack_user_id or None,
            workspace_id=workspace_id,
        )
    if tool == "describe_object":
        object_name = str(payload.get("object_name", "")).strip()
        return sf_describe_object(
            object_name,
            slack_user_id=slack_user_id or None,
            workspace_id=workspace_id,
        )
    if tool == "grep_artifact":
        artifact_id = str(payload.get("artifact_id", "")).strip()
        query = str(payload.get("query", "")).strip()
        max_hits = int(payload.get("max_hits", 40))
        return artifact_search_text(artifact_id=artifact_id, query=query, max_hits=max_hits)
    if tool == "kb_list":
        limit = int(payload.get("limit", 25))
        with SessionLocal() as db:
            items = list_knowledge_items(db=db, workspace_id=workspace_id, limit=max(1, min(limit, 200)))
            return {"count": len(items)}
    if tool == "kb_create":
        statement = str(payload.get("statement", "")).strip()
        title = str(payload.get("title", "")).strip() or statement[:120] or "Knowledge"
        kind_raw = str(payload.get("kind", "fact")).strip().lower()
        kind = _parse_kind(kind_raw)
        with SessionLocal() as db:
            create_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                kind=kind,
                confidence_tier=ConfidenceTier.coworker_context,
                confidence_score=0.8,
                title=title,
                content={"statement": statement, "source_type": "read_ingestion"},
                provenance={"pipeline": "read_ingestion_tools"},
            )
            db.commit()
        return {"created": True}
    if tool == "kb_update":
        knowledge_id = str(payload.get("knowledge_id", "")).strip()
        statement = _normalize_optional_text(payload.get("statement"))
        title = _normalize_optional_text(payload.get("title"))
        with SessionLocal() as db:
            item = update_knowledge_item(
                db=db,
                workspace_id=workspace_id,
                knowledge_id=knowledge_id,
                title=title,
                statement=statement,
            )
            db.commit()
        return {"updated": bool(item)}
    if tool == "kb_delete":
        knowledge_id = str(payload.get("knowledge_id", "")).strip()
        with SessionLocal() as db:
            deleted = delete_knowledge_item(db=db, workspace_id=workspace_id, knowledge_id=knowledge_id)
            db.commit()
        return {"deleted": bool(deleted)}
    raise ValueError(f"Unsupported ingestion tool: {tool}")


def _build_discovery_tool_specs(object_names: list[str]) -> list[DiscoveryToolSpec]:
    specs: list[DiscoveryToolSpec] = [
        DiscoveryToolSpec(
            key="validation_rules",
            tool_name="tool_validation_rules",
            reason="Capture validation rules and hard constraints.",
            api="tooling",
            query=(
                "SELECT Id, EntityDefinition.QualifiedApiName, ValidationName, Active, Description, "
                "ErrorMessage, ErrorDisplayField, ErrorConditionFormula FROM ValidationRule"
            ),
        ),
        DiscoveryToolSpec(
            key="naming_conventions",
            tool_name="tool_custom_objects",
            reason="Capture naming conventions and custom schema intent.",
            api="read",
            query="SELECT QualifiedApiName, Label FROM EntityDefinition WHERE IsCustomizable = true",
        ),
        DiscoveryToolSpec(
            key="automation_apex_tests",
            tool_name="tool_apex_tests",
            reason="Capture automation behavior from Apex test inventory.",
            api="tooling",
            query="SELECT Id, Name FROM ApexClass WHERE Name LIKE '%Test%'",
        ),
        DiscoveryToolSpec(
            key="automation_apex_triggers",
            tool_name="tool_apex_triggers",
            reason="Capture automation behavior from trigger inventory.",
            api="tooling",
            query="SELECT Id, Name, TableEnumOrId FROM ApexTrigger",
        ),
        DiscoveryToolSpec(
            key="automation_flows",
            tool_name="tool_flows",
            reason="Capture automation behavior from flow definitions.",
            api="tooling",
            query="SELECT Id, DeveloperName, ActiveVersion.VersionNumber FROM FlowDefinition",
        ),
    ]
    for object_name in object_names:
        slug = _object_key_slug(object_name)
        lit = _soql_literal(object_name)
        specs.append(
            DiscoveryToolSpec(
                key=f"describe_{object_name.lower()}",
                tool_name=f"tool_describe_{object_name.lower()}",
                reason=f"Describe object {object_name} and all fields.",
                api="describe",
                object_name=object_name,
            )
        )
        specs.append(
            DiscoveryToolSpec(
                key=f"object_policy_{slug}",
                tool_name=f"tool_object_policy_{slug}",
                reason=(
                    f"Tooling FieldDefinition policy audit for {object_name} "
                    "(required, restricted picklist, calculated, deprecated)."
                ),
                api="tooling",
                query=(
                    "SELECT QualifiedApiName, DataType, IsNillable, IsCalculated, IsRestrictedPicklist, "
                    f"IsDeprecatedAndHidden FROM FieldDefinition WHERE EntityDefinition.QualifiedApiName = {lit}"
                ),
                object_name=object_name,
            )
        )
        specs.append(
            DiscoveryToolSpec(
                key=f"object_validation_rules_{slug}",
                tool_name=f"tool_object_validation_rules_{slug}",
                reason=f"Tooling ValidationRule audit for {object_name}.",
                api="tooling",
                query=(
                    "SELECT ValidationName, Active, Description, ErrorMessage, ErrorDisplayField "
                    f"FROM ValidationRule WHERE EntityDefinition.QualifiedApiName = {lit}"
                ),
                object_name=object_name,
            )
        )
    return specs


def _extract_object_names_from_catalog(result: Any) -> list[str]:
    records = result.get("records", []) if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            api_name = str(item.get("QualifiedApiName", "")).strip()
            if api_name:
                names.append(api_name)
    for fallback in ["User", "Opportunity", "Account", "Lead", "Case"]:
        if fallback not in names:
            names.append(fallback)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _extend_preview(
    target: dict[str, list[dict[str, str]]],
    source: dict[str, list[dict[str, str]]],
) -> None:
    for key in ["facts", "hypotheses", "questions"]:
        target.setdefault(key, [])
        target[key].extend(source.get(key, []))


def _run_discovery_spec(
    *,
    spec: DiscoveryToolSpec,
    workspace_id: str,
    slack_user_id: str,
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None,
    parsed_intent: str,
    parsed_intent_reason: str,
) -> tuple[str, Any | None, str]:
    events.append(
        {
            "step": _next_step(events),
            "type": spec.tool_name,
            "status": "started",
            "reason": spec.reason,
        }
    )
    _emit_progress_update(
        progress_callback=progress_callback,
        events=events,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    try:
        if spec.api == "describe":
            result = _run_ingestion_tool(
                tool="describe_object",
                payload={"object_name": spec.object_name},
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
            )
        else:
            result = _run_ingestion_tool(
                tool="execute_soql",
                payload={"api": spec.api, "query": spec.query},
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
            )
        summary = _summarize_result_count(result, spec.key)
        context_text = _build_lossless_json_context(spec.key, result)
        artifact_note = _maybe_store_discovery_artifact(result=result, tool_name=spec.tool_name)
        artifact_id = _artifact_id_from_note(artifact_note)
        if artifact_note:
            summary = f"{summary}; {artifact_note}"
            context_text = f"{context_text}\n{artifact_note}"
        events[-1]["status"] = "success"
        events[-1]["reason"] = summary
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        return context_text, result, artifact_id
    except Exception as exc:
        if spec.api in {"read", "tooling"} and spec.query:
            repaired = _attempt_query_repair(
                query_repair={
                    "api": spec.api,
                    "query": spec.query,
                    "workspace_id": workspace_id,
                    "slack_user_id": slack_user_id,
                },
                error=exc,
                tool_name=spec.tool_name,
                max_attempts=DISCOVERY_REPAIR_MAX_ATTEMPTS,
            )
            if repaired is not None:
                result, repaired_query = repaired
                summary = _summarize_result_count(result, spec.key)
                context_text = _build_lossless_json_context(spec.key, result)
                artifact_note = _maybe_store_discovery_artifact(result=result, tool_name=spec.tool_name)
                artifact_id = _artifact_id_from_note(artifact_note)
                if artifact_note:
                    summary = f"{summary}; {artifact_note}"
                    context_text = f"{context_text}\n{artifact_note}"
                events[-1]["status"] = "success"
                events[-1]["reason"] = (
                    f"{summary}; repaired_query={_truncate_for_log(repaired_query, 180)}"
                )
                _emit_progress_update(
                    progress_callback=progress_callback,
                    events=events,
                    parsed_intent=parsed_intent,
                    parsed_intent_reason=parsed_intent_reason,
                )
                return context_text, result, artifact_id
        events[-1]["status"] = "error"
        events[-1]["reason"] = f"{spec.tool_name} failed: {type(exc).__name__}"
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        return f"{spec.tool_name} failed: {type(exc).__name__}", None, ""


def _artifact_id_from_note(artifact_note: str) -> str:
    text = str(artifact_note or "").strip()
    if text.startswith("artifact_id="):
        return text.split("=", 1)[1].strip()
    return ""


def _build_artifact_grep_context(
    *,
    artifact_id: str,
    workspace_id: str,
    slack_user_id: str,
) -> str:
    if not artifact_id:
        return ""
    queries = [
        "required",
        "picklist",
        "validation",
        "formula",
        "warning",
        "limit",
        "deprecated",
        "createable",
        "updateable",
    ]
    lines = [f"artifact_grep:{artifact_id}"]
    for query in queries:
        try:
            result = _run_ingestion_tool(
                tool="grep_artifact",
                payload={"artifact_id": artifact_id, "query": query, "max_hits": 25},
                workspace_id=workspace_id,
                slack_user_id=slack_user_id,
            )
        except Exception:
            continue
        hits = result.get("hits", []) if isinstance(result, dict) else []
        if not isinstance(hits, list) or not hits:
            continue
        rendered_hits = []
        for hit in hits[:12]:
            if not isinstance(hit, dict):
                continue
            path = str(hit.get("path", "")).strip()
            match = str(hit.get("match", "")).strip()
            if path:
                rendered_hits.append(f"{path}<{match or 'value'}>")
        if rendered_hits:
            lines.append(f"{query}: " + ", ".join(rendered_hits))
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def _persist_deterministic_tool_knowledge(
    *,
    workspace_id: str,
    spec: DiscoveryToolSpec,
    result: Any,
    validation_rule_names_by_object: dict[str, list[str]] | None = None,
) -> tuple[int, dict[str, list[dict[str, str]]]]:
    inserted = 0
    preview: dict[str, list[dict[str, str]]] = {"facts": [], "hypotheses": [], "questions": []}
    if not isinstance(result, dict):
        return inserted, preview
    with SessionLocal() as db:
        if spec.key == "validation_rules":
            records = result.get("records", [])
            if isinstance(records, list):
                for item in records:
                    if not isinstance(item, dict):
                        continue
                    title, statement, _, canonical_key = _validation_rule_knowledge_parts(item)
                    if not statement:
                        continue
                    resolve_or_supersede_by_canonical_key(
                        db=db,
                        workspace_id=workspace_id,
                        salesforce_org_key="default",
                        kind=KnowledgeKind.rule,
                        canonical_key=canonical_key,
                        replacement_content={"statement": statement, "source_type": "salesforce_metadata"},
                        replacement_title=title,
                        confidence_tier=ConfidenceTier.strict_violation,
                        confidence_score=0.98,
                        provenance={"pipeline": "read_ingestion_tools"},
                    )
                    inserted += 1
                    preview["facts"].append({"title": title, "statement": statement, "kind": "rule"})
        if spec.key.startswith("object_policy_") and spec.object_name:
            records = result.get("records", [])
            rec_list = records if isinstance(records, list) else []
            tot, req_ct, rpick, calc, dep = _summarize_field_definition_policy_counts(rec_list)
            statement = (
                f"Object `{spec.object_name}` FieldDefinition tooling policy summary: "
                f"field_count={tot}, required_field_count={req_ct} (IsNillable=false), "
                f"restricted_picklist_count={rpick}, calculated_field_count={calc}, "
                f"deprecated_hidden_field_count={dep}"
            )
            title = f"Object Policy Summary: {spec.object_name}"
            canonical_key = f"object_policy_summary:{spec.object_name}"
            resolve_or_supersede_by_canonical_key(
                db=db,
                workspace_id=workspace_id,
                salesforce_org_key="default",
                kind=KnowledgeKind.fact,
                canonical_key=canonical_key,
                replacement_content={"statement": statement, "source_type": "salesforce_metadata"},
                replacement_title=title,
                confidence_tier=ConfidenceTier.coworker_context,
                confidence_score=0.95,
                provenance={"pipeline": "read_ingestion_tools"},
            )
            inserted += 1
            preview["facts"].append({"title": title, "statement": statement, "kind": "fact"})
        if spec.key.startswith("object_validation_rules_") and spec.object_name:
            records = result.get("records", [])
            rec_list = records if isinstance(records, list) else []
            tot, act, inact = _summarize_validation_rule_inventory(rec_list)
            statement = (
                f"Object `{spec.object_name}` ValidationRule tooling summary: "
                f"total_rules={tot}, active_rules={act}, inactive_rules={inact}"
            )
            title = f"Object Validation Summary: {spec.object_name}"
            canonical_key = f"object_validation_summary:{spec.object_name}"
            resolve_or_supersede_by_canonical_key(
                db=db,
                workspace_id=workspace_id,
                salesforce_org_key="default",
                kind=KnowledgeKind.fact,
                canonical_key=canonical_key,
                replacement_content={"statement": statement, "source_type": "salesforce_metadata"},
                replacement_title=title,
                confidence_tier=ConfidenceTier.coworker_context,
                confidence_score=0.95,
                provenance={"pipeline": "read_ingestion_tools"},
            )
            inserted += 1
            preview["facts"].append({"title": title, "statement": statement, "kind": "fact"})
        if spec.api == "describe" and spec.object_name:
            fields = result.get("fields", [])
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    title, statement, canonical_key = _field_description_knowledge_parts(spec.object_name, field)
                    if not statement:
                        continue
                    resolve_or_supersede_by_canonical_key(
                        db=db,
                        workspace_id=workspace_id,
                        salesforce_org_key="default",
                        kind=KnowledgeKind.fact,
                        canonical_key=canonical_key,
                        replacement_content={"statement": statement, "source_type": "salesforce_metadata"},
                        replacement_title=title,
                        confidence_tier=ConfidenceTier.coworker_context,
                        confidence_score=0.95,
                        provenance={"pipeline": "read_ingestion_tools"},
                    )
                    inserted += 1
                    preview["facts"].append({"title": title, "statement": statement, "kind": "fact"})
            policy_title, policy_statement, policy_canonical_key = _object_policy_knowledge_parts(
                spec.object_name,
                result,
                (validation_rule_names_by_object or {}).get(spec.object_name, []),
            )
            if policy_statement:
                resolve_or_supersede_by_canonical_key(
                    db=db,
                    workspace_id=workspace_id,
                    salesforce_org_key="default",
                    kind=KnowledgeKind.rule,
                    canonical_key=policy_canonical_key,
                    replacement_content={"statement": policy_statement, "source_type": "salesforce_metadata"},
                    replacement_title=policy_title,
                    confidence_tier=ConfidenceTier.strict_violation,
                    confidence_score=0.95,
                    provenance={"pipeline": "read_ingestion_tools"},
                )
                inserted += 1
                preview["facts"].append(
                    {"title": policy_title, "statement": policy_statement, "kind": "rule"}
                )
        db.commit()
    return inserted, preview


def _run_knowledge_discovery(
    client: Any,
    settings: Settings,
    workspace_id: str,
    slack_user_id: str,
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None = None,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> tuple[dict[str, str], int, dict[str, list[dict[str, str]]]]:
    chunks: dict[str, str] = {}
    streamed_inserted = 0
    streamed_preview: dict[str, list[dict[str, str]]] = {"facts": [], "hypotheses": [], "questions": []}
    validation_rule_names_by_object: dict[str, list[str]] = {}
    object_catalog_spec = DiscoveryToolSpec(
        key="object_catalog",
        tool_name="tool_object_catalog",
        reason="List all present queryable objects to drive complete per-object auditing.",
        api="read",
        query=(
            "SELECT QualifiedApiName FROM EntityDefinition "
            "WHERE IsQueryable = true AND IsDeprecatedAndHidden = false"
        ),
    )
    catalog_context, catalog_result, _ = _run_discovery_spec(
        spec=object_catalog_spec,
        workspace_id=workspace_id,
        slack_user_id=slack_user_id,
        events=events,
        progress_callback=progress_callback,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    chunks[object_catalog_spec.key] = catalog_context
    object_names = _extract_object_names_from_catalog(catalog_result)
    discovery_specs = _build_discovery_tool_specs(object_names)

    for spec in discovery_specs:
        context_text, result, artifact_id = _run_discovery_spec(
            spec=spec,
            workspace_id=workspace_id,
            slack_user_id=slack_user_id,
            events=events,
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        chunks[spec.key] = context_text
        if spec.key == "validation_rules":
            validation_rule_names_by_object = _group_validation_rule_names_by_object(result)

        deterministic_added, deterministic_preview = _persist_deterministic_tool_knowledge(
            workspace_id=workspace_id,
            spec=spec,
            result=result,
            validation_rule_names_by_object=validation_rule_names_by_object,
        )
        streamed_inserted += deterministic_added
        _extend_preview(streamed_preview, deterministic_preview)

        if spec.api == "describe" and spec.object_name and isinstance(result, dict):
            policy_title, policy_statement, _ = _object_policy_knowledge_parts(
                spec.object_name,
                result,
                validation_rule_names_by_object.get(spec.object_name, []),
            )
            policy_chunk_key = f"policy_audit_{spec.object_name.lower()}"
            policy_chunk_text = (
                f"{policy_chunk_key}:\n"
                f"title={policy_title}\n"
                f"statement={policy_statement}"
            )
            chunks[policy_chunk_key] = policy_chunk_text
            policy_added, policy_preview = _extract_and_persist_chunk_knowledge(
                client=client,
                settings=settings,
                workspace_id=workspace_id,
                chunk_label=policy_chunk_key,
                chunk_text=policy_chunk_text,
            )
            streamed_inserted += policy_added
            _extend_preview(streamed_preview, policy_preview)

        llm_added, llm_preview = _extract_and_persist_chunk_knowledge(
            client=client,
            settings=settings,
            workspace_id=workspace_id,
            chunk_label=spec.key,
            chunk_text=context_text,
        )
        streamed_inserted += llm_added
        _extend_preview(streamed_preview, llm_preview)

        artifact_grep = _build_artifact_grep_context(
            artifact_id=artifact_id,
            workspace_id=workspace_id,
            slack_user_id=slack_user_id,
        )
        if artifact_grep:
            grep_key = f"{spec.key}_artifact_grep"
            chunks[grep_key] = artifact_grep
            grep_added, grep_preview = _extract_and_persist_chunk_knowledge(
                client=client,
                settings=settings,
                workspace_id=workspace_id,
                chunk_label=grep_key,
                chunk_text=artifact_grep,
            )
            streamed_inserted += grep_added
            _extend_preview(streamed_preview, grep_preview)
        logger.info(
            "Discovery step persisted: key=%s deterministic=%s llm=%s",
            spec.key,
            deterministic_added,
            llm_added,
        )

    return chunks, streamed_inserted, streamed_preview


def _extract_and_persist_chunk_knowledge(
    client: Any,
    settings: Settings,
    workspace_id: str,
    chunk_label: str,
    chunk_text: str,
) -> tuple[int, dict[str, list[dict[str, str]]]]:
    if not chunk_text or " failed:" in chunk_text.lower():
        return 0, {"facts": [], "hypotheses": [], "questions": []}
    system_prompt = """
You extract structured knowledge from one Salesforce metadata chunk.
Return strict JSON only:
{
  "facts":[{"title":"...","statement":"...","kind":"fact|rule","confidence_tier":"strict_violation|coworker_context","confidence_score":0.0,"sf_object_api_name":"Opportunity|null"}],
  "hypotheses":[{"title":"...","statement":"...","confidence_tier":"coworker_context","confidence_score":0.0}],
  "questions":[{"title":"...","question":"...","why_needed":"...","blocking_policy":true}]
}
Rules:
- Include as many concrete policies/constraints as present.
- Extraction priority in this chunk:
  1) explicit validation/constraint statements
  2) object/field semantics (required, type, picklist, writeability)
  3) naming and custom-schema intent
  4) automation implications
- Include object/field API names whenever available.
- No markdown, no prose outside JSON.
"""
    payload = f"chunk_label={chunk_label}\n\nchunk_data:\n{chunk_text}"
    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1200,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": payload}],
        )
        raw = extract_text_response(response)
        doc = extract_json_object(raw)
        inserted, preview = _persist_ingestion_document(
            workspace_id=workspace_id,
            document=doc,
            max_items=1000000,
        )
        return inserted, preview
    except Exception as exc:
        logger.info("Chunk-level extraction/persist failed for %s: %s", chunk_label, exc)
        return 0, {"facts": [], "hypotheses": [], "questions": []}


def _attempt_query_repair(
    query_repair: dict[str, str],
    error: Exception,
    tool_name: str,
    max_attempts: int = 1,
) -> tuple[dict[str, Any], str] | None:
    api = str(query_repair.get("api", "")).strip().lower()
    original_query = str(query_repair.get("query", "")).strip()
    workspace_id = str(query_repair.get("workspace_id", "")).strip()
    slack_user_id = str(query_repair.get("slack_user_id", "")).strip()
    if api not in {"tooling", "read"} or not original_query:
        return None
    client = get_claude_client()
    attempts = max(1, int(max_attempts))
    failed_query = original_query
    last_error: Exception = error
    for attempt in range(1, attempts + 1):
        try:
            repair_input = (
                f"tool={tool_name}\n"
                f"api={api}\n"
                f"error={type(last_error).__name__}: {last_error}\n"
                f"failed_query={failed_query}\n"
                "Return only corrected read-only SELECT SOQL."
            )
            repaired_payload = repair_json_object_with_llm(
                client=client,
                model="claude-3-5-haiku-20241022",
                raw_text=repair_input,
                schema_hint='{"query":"SELECT ..."} read-only SOQL preserving original intent',
                max_tokens=700,
            )
            if repaired_payload is None:
                raise RuntimeError("repair returned no query payload")
            repaired_query = str(repaired_payload.get("query", "")).strip()
            repaired_query = ensure_read_only_select(repaired_query, context="SOQL")
            if api == "tooling":
                result = sf_tooling_query(
                    repaired_query,
                    slack_user_id=slack_user_id or None,
                    workspace_id=workspace_id or None,
                )
                return result, repaired_query
            result = sf_query_read_only(
                repaired_query,
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id or None,
            )
            return result, repaired_query
        except Exception as repair_exc:
            last_error = repair_exc
            if "repaired_query" in locals() and str(locals()["repaired_query"]).strip():
                failed_query = str(locals()["repaired_query"]).strip()
            logger.info(
                "SOQL repair attempt %s/%s failed for %s: %s",
                attempt,
                attempts,
                tool_name,
                repair_exc,
            )
            continue
    return None


def _summarize_result_count(result: dict[str, Any], label: str) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    count = len(records) if isinstance(records, list) else 0
    return f"{label}={count}"


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    return text


def _next_step(events: list[dict[str, str]]) -> str:
    return str(len(events) + 1)


def _build_persisted_items_digest(persisted: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = ["Persisted knowledge items:"]
    facts = persisted.get("facts", [])
    hypotheses = persisted.get("hypotheses", [])
    questions = persisted.get("questions", [])
    if facts:
        lines.append(f"Facts / Rules (showing first 5 of {len(facts)}):")
        for item in facts[:5]:
            kind = str(item.get("kind", "fact")).strip()
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            statement = str(item.get("statement", "")).strip()
            if statement:
                lines.append(f"- [{kind}] {title}: {statement}")
    if hypotheses:
        lines.append(f"Hypotheses (showing first 5 of {len(hypotheses)}):")
        for item in hypotheses[:5]:
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            statement = str(item.get("statement", "")).strip()
            if statement:
                lines.append(f"- {title}: {statement}")
    if questions:
        lines.append(f"Questions (showing first 5 of {len(questions)}):")
        for item in questions[:5]:
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            question = str(item.get("question", "")).strip()
            if question:
                lines.append(f"- {title}: {question}")
    if len(lines) == 1:
        lines.append("- No facts, hypotheses, or questions were persisted.")
    return "\n".join(lines)


def _truncate_for_log(text: str, max_chars: int = INGESTION_LOG_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... [truncated log output, omitted {omitted} chars]"


def _build_observability_blob(
    events: list[dict[str, str]],
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    execution_plan: list[str] | None = None,
) -> str:
    lines = ["Execution trace"]
    if parsed_intent:
        lines.append(f"- Intent parse: {parsed_intent}")
    if parsed_intent_reason:
        lines.append(f"- Intent reason: {parsed_intent_reason}")
    if execution_plan:
        lines.append("- Execution plan:")
        for idx, step in enumerate(execution_plan, start=1):
            lines.append(f"  - {idx}. {step}")
    if not events:
        lines.append("- No ingestion steps were executed.")
        return "```\n" + "\n".join(lines) + "\n```"
    for event in events:
        lines.append(
            f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


def _emit_progress_update(
    progress_callback: Callable[[str], None] | None,
    events: list[dict[str, str]],
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(
            _build_observability_blob(
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
                execution_plan=_ingestion_execution_plan(),
            )
        )
    except Exception as exc:
        logger.info("Could not emit ingestion progress update: %s", exc)


def _ingestion_execution_plan() -> list[str]:
    return [
        "Build metadata-first discovery scope (object catalog + describe coverage).",
        "Run per-object policy audits (Tooling FieldDefinition + ValidationRule per object) alongside describe.",
        "Execute centralized ingestion tools (SOQL, describe, artifact grep).",
        "Extract and persist knowledge incrementally after each tool output.",
        "Upsert deterministic rule/field knowledge for validation, describe, and policy-summary results.",
        "Return human-readable persisted output.",
    ]


def _maybe_store_discovery_artifact(result: Any, tool_name: str) -> str:
    try:
        rendered = json.dumps(result, ensure_ascii=True)
    except Exception:
        return ""
    if len(rendered) <= DISCOVERY_ARTIFACT_THRESHOLD_CHARS:
        return ""
    try:
        stored = artifact_store(payload=result, source=f"knowledge_discovery:{tool_name}")
        artifact_id = str(stored.get("artifact_id", "")).strip()
        if artifact_id:
            return f"artifact_id={artifact_id}"
    except Exception as exc:
        logger.info("Could not store discovery artifact for %s: %s", tool_name, exc)
    return ""

