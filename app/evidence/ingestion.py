from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable

from app.agent.tools import artifact_store, sf_describe_object, sf_query_read_only, sf_tooling_query
from app.config import Settings
from app.db.enums import ConfidenceTier, KnowledgeKind, KnowledgeQuestionStatus
from app.db.repository import create_knowledge_item
from app.db.session import SessionLocal
from app.llm.client import get_claude_client

logger = logging.getLogger(__name__)
INGESTION_LOG_PREVIEW_CHARS = 12000
DISCOVERY_QUERY_LIMIT = 200
DISCOVERY_ARTIFACT_THRESHOLD_CHARS = 5000
TOOL_TRACE_MAX_STEPS = 80


@dataclass
class IngestionRunResult:
    status: str
    message: str
    observability: str

INGESTION_PROMPT = """
You extract structured Salesforce knowledge from a completed read-agent response.

Return ONLY valid JSON matching this shape:
{
  "facts": [
    {
      "title": "...",
      "statement": "...",
      "kind": "fact|rule|trend",
      "source": "validation_rule|feature_behavior|naming_convention|automation_test",
      "confidence_tier": "observed_trend|strict_violation",
      "confidence_score": 0.0,
      "sf_object_api_name": "Opportunity|null"
    }
  ],
  "hypotheses": [
    {
      "title": "...",
      "statement": "...",
      "confidence_tier": "similar_past_approval|observed_trend",
      "confidence_score": 0.0
    }
  ],
  "questions": [
    {
      "title": "...",
      "question": "...",
      "why_needed": "...",
      "blocking_policy": true
    }
  ]
}

Rules:
- Do not invent unsupported Salesforce facts.
- Prefer concrete, atomic statements.
- Keep confidence_score between 0 and 1.
- For strict validation-rule-style constraints use confidence_tier=strict_violation.
- For inferred behavior use confidence_tier=observed_trend or similar_past_approval.
- Keep output compact and parseable. Use short titles and short statements.
- Hard limits:
  - facts: at most 12
  - hypotheses: at most 6
  - questions: exactly 3
- Priority order for fact extraction (highest first):
  1) Examine validation rules and hard constraints found in salesforce. Establish facts for each.
  2) Examine feature behavior and object/field descriptions.
  3) Review naming conventions and schema semantics.
  4) Collect automation/test evidence (Apex tests, flows, triggers, coverage cues).
- Omit operational app/workflow history entirely; learn from Salesforce instance metadata only.
- Avoid "observed trend" style statements for now.
- Questions should be specific to the current workspace.
- Avoid questions that are related to the approval process or execution of plans.
- Known policy (fixed): only the designated human co-worker can approve/auto-approve tasks; end-users cannot.
- Do not generate facts, hypotheses, or questions that challenge or re-investigate this approval policy.
- Prioritize asking questions that target policy, constraints, required inputs, and edge-case behavior.
- Generate exactly 1 high-quality question from the highest-value hypothesis/uncertainty gap.
- If validation-rule details are present in discovery context, convert them into concrete `kind="rule"` facts.
- Do not ask questions that simply request the content of validation rules already provided in context.
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
    discovery_contexts = _run_knowledge_discovery(
        settings=settings,
        workspace_id=workspace_id,
        slack_user_id=slack_user_id,
        events=events,
        progress_callback=progress_callback,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    stage_doc = _extract_knowledge_in_stages(
        client=client,
        settings=settings,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
        user_text=user_text,
        discovery_contexts=discovery_contexts,
        events=events,
        progress_callback=progress_callback,
    )
    if stage_doc is None:
        obs = _build_observability_blob(
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
            execution_plan=_ingestion_execution_plan(),
        )
        return IngestionRunResult(
            status="error",
            message=f"Knowledge ingestion JSON parse failed. Please retry.\n\n{obs}",
            observability=obs,
        )
    try:
        events.append(
            {
                "step": _next_step(events),
                "type": "persist_knowledge_items",
                "status": "started",
                "reason": "Persist extracted knowledge items into database.",
            }
        )
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        inserted, persisted_preview = _persist_ingestion_document(
            workspace_id=workspace_id,
            document=stage_doc,
            max_items=settings.knowledge_ingestion_max_items,
        )
        events[-1]["status"] = "success"
        events[-1]["reason"] = f"Persisted {inserted} knowledge items."
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
    fact_cap, hypothesis_cap, question_cap = _ingestion_caps(max_items)
    fact_inserted = 0
    hypothesis_inserted = 0
    question_inserted = 0
    with SessionLocal() as db:
        for item in facts:
            if inserted >= max_items or fact_inserted >= fact_cap:
                break
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
            fact_inserted += 1
            persisted_preview["facts"].append(
                {
                    "title": title or "Extracted fact",
                    "statement": statement,
                    "kind": kind.value,
                }
            )

        for item in questions:
            if inserted >= max_items or question_inserted >= question_cap:
                break
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
            question_inserted += 1
            persisted_preview["questions"].append(
                {
                    "title": title or "Question",
                    "question": question,
                    "why_needed": str(item.get("why_needed", "")).strip(),
                }
            )

        for item in hypotheses:
            if inserted >= max_items or hypothesis_inserted >= hypothesis_cap:
                break
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
            hypothesis_inserted += 1
            persisted_preview["hypotheses"].append(
                {
                    "title": title or "Hypothesis",
                    "statement": statement,
                }
            )
        db.commit()
    return inserted, persisted_preview


def _ingestion_caps(max_items: int) -> tuple[int, int, int]:
    budget = max(1, max_items)
    facts = min(12, max(4, budget // 2))
    questions = 1
    hypotheses = min(6, max(1, budget - facts - questions))
    return facts, hypotheses, questions


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


def _run_knowledge_discovery(
    settings: Settings,
    workspace_id: str,
    slack_user_id: str,
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None = None,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> dict[str, str]:
    chunks: dict[str, str] = {}

    # 1) Validation rules and constraints
    chunks["validation_rules"] = (
        _call_discovery_tool(
            events=events,
            tool_name="tool_validation_rules",
            reason="Look up active validation rules and constraints first.",
            call=lambda: sf_tooling_query(
                (
                    "SELECT Id, EntityDefinition.QualifiedApiName, "
                    "ValidationName, Active, Description, ErrorMessage "
                    "FROM ValidationRule LIMIT 80"
                ),
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            ),
            summarize=_summarize_validation_rules,
            context_builder=_build_validation_rules_context,
            query_repair={
                "api": "tooling",
                "query": (
                    "SELECT Id, EntityDefinition.QualifiedApiName, "
                    "ValidationName, Active, Description, ErrorMessage "
                    "FROM ValidationRule LIMIT 80"
                ),
                "workspace_id": workspace_id,
                "slack_user_id": slack_user_id,
            },
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
    )

    # 2) Feature behavior and object/field descriptions
    for obj in ["Opportunity", "Account", "Lead", "Case"]:
        chunks[f"describe_{obj.lower()}"] = (
            _call_discovery_tool(
                events=events,
                tool_name=f"tool_describe_{obj.lower()}",
                reason=f"Describe {obj} behavior and field semantics.",
                call=lambda obj_name=obj: sf_describe_object(
                    obj_name,
                    slack_user_id=slack_user_id or None,
                    workspace_id=workspace_id,
                ),
                summarize=lambda result, obj_name=obj: _summarize_describe(result, obj_name),
                context_builder=lambda result, obj_name=obj: _build_describe_context(result, obj_name),
                progress_callback=progress_callback,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
        )

    # 3) Naming conventions
    chunks["naming_conventions"] = (
        _call_discovery_tool(
            events=events,
            tool_name="tool_custom_objects",
            reason="Review object naming conventions from custom objects.",
            call=lambda: sf_query_read_only(
                "SELECT QualifiedApiName, Label FROM EntityDefinition "
                f"WHERE IsCustomizable = true LIMIT {DISCOVERY_QUERY_LIMIT}",
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            ),
            summarize=_summarize_entity_definition_names,
            context_builder=_build_naming_conventions_context,
            query_repair={
                "api": "read",
                "query": (
                    "SELECT QualifiedApiName, Label FROM EntityDefinition "
                    f"WHERE IsCustomizable = true LIMIT {DISCOVERY_QUERY_LIMIT}"
                ),
                "workspace_id": workspace_id,
                "slack_user_id": slack_user_id,
            },
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
    )

    # 4) Tests, flows, triggers
    chunks["automation_apex_tests"] = (
        _call_discovery_tool(
            events=events,
            tool_name="tool_apex_tests",
            reason="Review Apex test inventory.",
            call=lambda: sf_tooling_query(
                f"SELECT Id, Name FROM ApexClass WHERE Name LIKE '%Test%' LIMIT {DISCOVERY_QUERY_LIMIT}",
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            ),
            summarize=lambda result: _summarize_name_records(result, label="apex_tests"),
            context_builder=lambda result: _build_name_records_context(result, "apex_tests"),
            query_repair={
                "api": "tooling",
                "query": f"SELECT Id, Name FROM ApexClass WHERE Name LIKE '%Test%' LIMIT {DISCOVERY_QUERY_LIMIT}",
                "workspace_id": workspace_id,
                "slack_user_id": slack_user_id,
            },
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
    )
    chunks["automation_apex_triggers"] = (
        _call_discovery_tool(
            events=events,
            tool_name="tool_apex_triggers",
            reason="Review Apex trigger inventory.",
            call=lambda: sf_tooling_query(
                f"SELECT Id, Name, TableEnumOrId FROM ApexTrigger LIMIT {DISCOVERY_QUERY_LIMIT}",
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            ),
            summarize=lambda result: _summarize_name_records(result, label="apex_triggers"),
            context_builder=lambda result: _build_name_records_context(result, "apex_triggers"),
            query_repair={
                "api": "tooling",
                "query": f"SELECT Id, Name, TableEnumOrId FROM ApexTrigger LIMIT {DISCOVERY_QUERY_LIMIT}",
                "workspace_id": workspace_id,
                "slack_user_id": slack_user_id,
            },
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
    )
    chunks["automation_flows"] = (
        _call_discovery_tool(
            events=events,
            tool_name="tool_flows",
            reason="Review active flow definitions.",
            call=lambda: sf_tooling_query(
                "SELECT Id, DeveloperName, ActiveVersion.VersionNumber "
                f"FROM FlowDefinition LIMIT {DISCOVERY_QUERY_LIMIT}",
                slack_user_id=slack_user_id or None,
                workspace_id=workspace_id,
            ),
            summarize=_summarize_flows,
            context_builder=_build_flows_context,
            query_repair={
                "api": "tooling",
                "query": (
                    "SELECT Id, DeveloperName, ActiveVersion.VersionNumber "
                    f"FROM FlowDefinition LIMIT {DISCOVERY_QUERY_LIMIT}"
                ),
                "workspace_id": workspace_id,
                "slack_user_id": slack_user_id,
            },
            progress_callback=progress_callback,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
    )

    return chunks


def _extract_knowledge_in_stages(
    client: Any,
    settings: Settings,
    parsed_intent: str,
    parsed_intent_reason: str,
    user_text: str,
    discovery_contexts: dict[str, str],
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    stage_plan = [
        ("validation_rules", "validation_rule", 1200),
        ("feature_behavior", "feature_behavior", 1000),
        ("naming_conventions", "naming_convention", 900),
        ("automation_tests", "automation_test", 1000),
    ]
    merged: dict[str, Any] = {"facts": [], "hypotheses": [], "questions": []}
    parsed_any = False
    for stage_name, source_label, stage_tokens in stage_plan:
        context = _compose_stage_context(stage_name=stage_name, discovery_contexts=discovery_contexts)
        prompt = _build_stage_ingestion_prompt(
            source_label=source_label,
            allow_question=stage_name == "automation_tests",
        )
        payload = (
            "Request context:\n"
            f"- parsed_intent: {parsed_intent}\n"
            f"- parsed_intent_reason: {parsed_intent_reason}\n"
            f"- user_text: {user_text}\n"
            f"- stage: {stage_name}\n\n"
            "Discovery context:\n"
            f"{context}\n\n"
            "Instructions:\n"
            "- Learn from Salesforce metadata context only.\n"
            "- Do not include app workflow/approval lifecycle content.\n"
        )
        events.append(
            {
                "step": _next_step(events),
                "type": f"extract_{stage_name}",
                "status": "started",
                "reason": f"Extract stage-focused knowledge for {stage_name}.",
            }
        )
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=stage_tokens,
                temperature=0,
                system=prompt,
                messages=[{"role": "user", "content": payload}],
            )
            events[-1]["status"] = "success"
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
        except Exception as exc:
            events[-1]["status"] = "error"
            events[-1]["reason"] = f"LLM stage failed: {type(exc).__name__}"
            logger.info("Knowledge ingestion stage call failed (%s): %s", stage_name, exc)
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            continue
        raw = _extract_text_response(response)
        events.append(
            {
                "step": _next_step(events),
                "type": f"parse_{stage_name}",
                "status": "started",
                "reason": "Parse stage JSON payload.",
            }
        )
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        try:
            doc = _extract_json_object(raw)
            _merge_ingestion_docs(merged, doc)
            parsed_any = True
            events[-1]["status"] = "success"
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
        except Exception as exc:
            repaired = _retry_stage_parse_repair(
                client=client,
                settings=settings,
                raw=raw,
                stage_name=stage_name,
                events=events,
                progress_callback=progress_callback,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            if repaired is not None:
                _merge_ingestion_docs(merged, repaired)
                parsed_any = True
                events[-1]["status"] = "success"
                events[-1]["reason"] = "Stage JSON parsed after repair retry."
                _emit_progress_update(
                    progress_callback=progress_callback,
                    events=events,
                    parsed_intent=parsed_intent,
                    parsed_intent_reason=parsed_intent_reason,
                )
                continue
            events[-1]["status"] = "error"
            events[-1]["reason"] = "Stage JSON parse failed."
            logger.info("Knowledge ingestion JSON parse failed for stage=%s: %s", stage_name, exc)
            logger.info(
                "Knowledge ingestion raw stage output (%s, len=%s):\n%s",
                stage_name,
                len(raw),
                _truncate_for_log(raw, max_chars=INGESTION_LOG_PREVIEW_CHARS),
            )
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            continue
    if not parsed_any:
        return None
    return merged


def _extract_text_response(response: Any) -> str:
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    return "\n".join(text_parts).strip()


def _compose_stage_context(stage_name: str, discovery_contexts: dict[str, str]) -> str:
    if stage_name == "validation_rules":
        return discovery_contexts.get("validation_rules", "")
    if stage_name == "feature_behavior":
        return "\n".join(
            [
                discovery_contexts.get("describe_opportunity", ""),
                discovery_contexts.get("describe_account", ""),
                discovery_contexts.get("describe_lead", ""),
            ]
        ).strip()
    if stage_name == "naming_conventions":
        return discovery_contexts.get("naming_conventions", "")
    if stage_name == "automation_tests":
        return "\n".join(
            [
                discovery_contexts.get("automation_apex_tests", ""),
                discovery_contexts.get("automation_apex_triggers", ""),
                discovery_contexts.get("automation_flows", ""),
            ]
        ).strip()
    return ""


def _build_stage_ingestion_prompt(source_label: str, allow_question: bool) -> str:
    question_instruction = (
        "- questions: exactly 1 high-value gap question." if allow_question else "- questions: [] (none)."
    )
    return f"""
You extract structured Salesforce knowledge from a completed read-agent response.

Return ONLY valid JSON matching this shape:
{{
  "facts": [
    {{
      "title": "...",
      "statement": "...",
      "kind": "fact|rule",
      "source": "{source_label}",
      "confidence_tier": "strict_violation|coworker_context",
      "confidence_score": 0.0,
      "sf_object_api_name": "Opportunity|null"
    }}
  ],
  "hypotheses": [
    {{
      "title": "...",
      "statement": "...",
      "confidence_tier": "strict_violation|coworker_context",
      "confidence_score": 0.0
    }}
  ],
  "questions": [
    {{
      "title": "...",
      "question": "...",
      "why_needed": "...",
      "blocking_policy": true
    }}
  ]
}}

Rules:
- Keep output compact and parseable.
- facts: at most 4
- hypotheses: at most 2
{question_instruction}
- Do not invent unsupported facts.
- Prefer durable rules and constraints over timeline narration.
- Treat validation rules, field requirements, picklist semantics, and automation evidence as primary facts.
- Omit app workflow/approval lifecycle facts.
- Avoid observed-trend framing.
- Known policy (fixed): only the designated human co-worker can approve/auto-approve tasks; end-users cannot.
- Do not generate facts, hypotheses, or questions that challenge or re-investigate this approval policy.
- Keep confidence_score between 0 and 1.
- If validation-rule details exist, convert them to kind=rule facts.
- If rule details are present, do not ask questions requesting those same rule details.
"""


def _merge_ingestion_docs(base: dict[str, Any], new_doc: dict[str, Any]) -> None:
    for key in ["facts", "hypotheses", "questions"]:
        items = new_doc.get(key, [])
        if not isinstance(items, list):
            continue
        if key not in base or not isinstance(base[key], list):
            base[key] = []
        base[key].extend([item for item in items if isinstance(item, dict)])


def _retry_stage_parse_repair(
    client: Any,
    settings: Settings,
    raw: str,
    stage_name: str,
    events: list[dict[str, str]],
    progress_callback: Callable[[str], None] | None,
    parsed_intent: str,
    parsed_intent_reason: str,
) -> dict[str, Any] | None:
    events.append(
        {
            "step": _next_step(events),
            "type": f"repair_{stage_name}",
            "status": "started",
            "reason": "Repair malformed stage JSON using error-guided rewrite.",
        }
    )
    _emit_progress_update(
        progress_callback=progress_callback,
        events=events,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    repair_prompt = (
        "You are a strict JSON repair tool. Convert the provided malformed output into valid JSON "
        "with exactly keys: facts (array), hypotheses (array), questions (array). "
        "Return JSON only, no markdown, no prose."
    )
    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1000,
            temperature=0,
            system=repair_prompt,
            messages=[{"role": "user", "content": raw[:7000]}],
        )
        repaired_raw = _extract_text_response(response)
        repaired_doc = _extract_json_object(repaired_raw)
        events[-1]["status"] = "success"
        events[-1]["reason"] = "Repair retry returned valid JSON."
        return repaired_doc
    except Exception as exc:
        events[-1]["status"] = "error"
        events[-1]["reason"] = f"Repair retry failed: {type(exc).__name__}"
        logger.info("Knowledge ingestion JSON repair retry failed for stage=%s: %s", stage_name, exc)
        return None
    finally:
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )


def _call_discovery_tool(
    events: list[dict[str, str]],
    tool_name: str,
    reason: str,
    call: Any,
    summarize: Any,
    context_builder: Any | None = None,
    retry_calls: list[tuple[str, Any]] | None = None,
    query_repair: dict[str, str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> str:
    events.append(
        {
            "step": _next_step(events),
            "type": tool_name,
            "status": "started",
            "reason": reason,
        }
    )
    _emit_progress_update(
        progress_callback=progress_callback,
        events=events,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    attempts: list[tuple[str, Any]] = [("primary", call)]
    if retry_calls:
        attempts.extend(retry_calls)
    last_exc: Exception | None = None
    for idx, (attempt_name, attempt_call) in enumerate(attempts):
        try:
            result = attempt_call()
            summary = summarize(result)
            context_text = context_builder(result) if context_builder is not None else summary
            artifact_note = _maybe_store_discovery_artifact(result=result, tool_name=tool_name)
            if artifact_note:
                summary = f"{summary}; {artifact_note}"
                context_text = f"{context_text}; {artifact_note}"
            if idx > 0:
                summary = f"{summary}; retry={attempt_name}"
            events[-1]["status"] = "success"
            events[-1]["reason"] = summary
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            return f"{tool_name}: {context_text}"
        except Exception as exc:
            last_exc = exc
            if idx < len(attempts) - 1:
                events[-1]["reason"] = (
                    f"{tool_name} attempt failed ({type(exc).__name__}); retrying with {attempts[idx + 1][0]}"
                )
                _emit_progress_update(
                    progress_callback=progress_callback,
                    events=events,
                    parsed_intent=parsed_intent,
                    parsed_intent_reason=parsed_intent_reason,
                )
                continue
            if query_repair is not None:
                repaired = _attempt_query_repair(query_repair=query_repair, error=exc, tool_name=tool_name)
                if repaired is not None:
                    result, repaired_query = repaired
                    summary = summarize(result)
                    context_text = context_builder(result) if context_builder is not None else summary
                    artifact_note = _maybe_store_discovery_artifact(result=result, tool_name=tool_name)
                    if artifact_note:
                        summary = f"{summary}; {artifact_note}"
                        context_text = f"{context_text}; {artifact_note}"
                    events[-1]["status"] = "success"
                    events[-1]["reason"] = f"{summary}; repaired_query={_truncate_for_log(repaired_query, 180)}"
                    _emit_progress_update(
                        progress_callback=progress_callback,
                        events=events,
                        parsed_intent=parsed_intent,
                        parsed_intent_reason=parsed_intent_reason,
                    )
                    return f"{tool_name}: {context_text}"
    msg = f"{tool_name} failed: {type(last_exc).__name__ if last_exc else 'UnknownError'}"
    events[-1]["status"] = "error"
    events[-1]["reason"] = msg
    _emit_progress_update(
        progress_callback=progress_callback,
        events=events,
        parsed_intent=parsed_intent,
        parsed_intent_reason=parsed_intent_reason,
    )
    return msg


def _attempt_query_repair(
    query_repair: dict[str, str],
    error: Exception,
    tool_name: str,
) -> tuple[dict[str, Any], str] | None:
    api = str(query_repair.get("api", "")).strip().lower()
    original_query = str(query_repair.get("query", "")).strip()
    workspace_id = str(query_repair.get("workspace_id", "")).strip()
    slack_user_id = str(query_repair.get("slack_user_id", "")).strip()
    if api not in {"tooling", "read"} or not original_query:
        return None
    try:
        client = get_claude_client()
        repair_prompt = (
            "You are a SOQL repair helper. Given a failed query and error, return ONLY JSON "
            "with shape {\"query\":\"SELECT ...\"}. Keep it read-only and valid for the same intent."
        )
        repair_input = (
            f"tool={tool_name}\n"
            f"api={api}\n"
            f"error={type(error).__name__}: {error}\n"
            f"failed_query={original_query}\n"
            "Return only corrected read-only SELECT SOQL."
        )
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=600,
            temperature=0,
            system=repair_prompt,
            messages=[{"role": "user", "content": repair_input}],
        )
        repaired_raw = _extract_text_response(response)
        repaired_payload = _extract_json_object(repaired_raw)
        repaired_query = str(repaired_payload.get("query", "")).strip()
        if not repaired_query.lower().startswith("select"):
            return None
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
        logger.info("SOQL repair attempt failed for %s: %s", tool_name, repair_exc)
        return None


def _summarize_validation_rules(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    count = len(records) if isinstance(records, list) else 0
    return f"validation_rules={count}"


def _build_validation_rules_context(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list) or not records:
        return "validation_rules=0"
    lines = [f"validation_rules={len(records)}"]
    for item in records[:50]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("ValidationName", "")).strip() or "UnnamedRule"
        display_field = str(item.get("ErrorDisplayField", "")).strip() or "<none>"
        message = str(item.get("ErrorMessage", "")).strip() or "<none>"
        formula = str(item.get("ErrorConditionFormula", "")).strip()
        object_name = "<unknown>"
        entity = item.get("EntityDefinition")
        if isinstance(entity, dict):
            object_name = str(entity.get("QualifiedApiName", "")).strip() or "<unknown>"
        formula_preview = formula.replace("\n", " ").strip()
        if len(formula_preview) > 160:
            formula_preview = formula_preview[:157] + "..."
        if len(message) > 140:
            message = message[:137] + "..."
        lines.append(
            f"- rule={name}; object={object_name}; display_field={display_field}; "
            f"message={message}; formula={formula_preview or '<none>'}"
        )
    return "\n".join(lines)


def _summarize_describe(result: dict[str, Any], object_name: str) -> str:
    fields = result.get("fields", []) if isinstance(result, dict) else []
    field_count = len(fields) if isinstance(fields, list) else 0
    return f"{object_name} fields={field_count}"


def _build_describe_context(result: dict[str, Any], object_name: str) -> str:
    if not isinstance(result, dict):
        return f"{object_name}: describe unavailable"
    fields = result.get("fields", [])
    if not isinstance(fields, list):
        fields = []
    required_fields: list[str] = []
    picklist_notes: list[str] = []
    numeric_constraints: list[str] = []
    naming_examples: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        api_name = str(field.get("name", "")).strip()
        label = str(field.get("label", "")).strip()
        if not api_name:
            continue
        if not bool(field.get("nillable", True)) and not bool(field.get("defaultedOnCreate", False)):
            required_fields.append(api_name)
        field_type = str(field.get("type", "")).strip().lower()
        if field_type == "picklist":
            values = field.get("picklistValues", [])
            picks: list[str] = []
            if isinstance(values, list):
                for val in values[:8]:
                    if isinstance(val, dict) and bool(val.get("active", True)):
                        picks.append(str(val.get("value", "")).strip())
            if picks:
                picklist_notes.append(f"{api_name}={','.join([p for p in picks if p])}")
        if field_type in {"currency", "double", "percent", "int"}:
            precision = field.get("precision")
            scale = field.get("scale")
            if precision is not None:
                numeric_constraints.append(f"{api_name}(precision={precision},scale={scale})")
        if label and api_name.endswith("__c"):
            naming_examples.append(f"{api_name}:{label}")

    lines = [
        f"{object_name} describe:",
        f"- required_fields: {', '.join(required_fields[:20]) or '<none>'}",
        f"- picklists: {'; '.join(picklist_notes[:10]) or '<none>'}",
        f"- numeric_constraints: {'; '.join(numeric_constraints[:10]) or '<none>'}",
        f"- custom_field_naming_examples: {'; '.join(naming_examples[:12]) or '<none>'}",
    ]
    return "\n".join(lines)


def _summarize_entity_definition_names(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(records, list):
        for item in records[:25]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("QualifiedApiName", "")).strip()
            if name:
                names.append(name)
    return f"custom_objects={len(records) if isinstance(records, list) else 0}; sample={','.join(names)}"


def _build_naming_conventions_context(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        return "naming_conventions: unavailable"
    custom_suffix_count = 0
    title_case_like = 0
    examples: list[str] = []
    for item in records[:80]:
        if not isinstance(item, dict):
            continue
        api_name = str(item.get("QualifiedApiName", "")).strip()
        label = str(item.get("Label", "")).strip()
        if not api_name:
            continue
        if api_name.endswith("__c"):
            custom_suffix_count += 1
        if label and label[:1].isupper():
            title_case_like += 1
        if len(examples) < 20:
            examples.append(f"{api_name}:{label}")
    total = len(records)
    return (
        "naming_conventions:\n"
        f"- total_objects: {total}\n"
        f"- custom_suffix_objects: {custom_suffix_count}\n"
        f"- title_case_label_ratio: {title_case_like}/{max(total, 1)}\n"
        f"- object_examples: {'; '.join(examples) or '<none>'}"
    )


def _summarize_name_records(result: dict[str, Any], label: str) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(records, list):
        for item in records[:25]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("Name", "")).strip()
            if name:
                names.append(name)
    return f"{label}={len(records) if isinstance(records, list) else 0}; sample={','.join(names)}"


def _build_name_records_context(result: dict[str, Any], label: str) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        return f"{label}: unavailable"
    names: list[str] = []
    for item in records[:50]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name", "")).strip()
        if name:
            names.append(name)
    return f"{label}:\n- count: {len(records)}\n- names: {', '.join(names) or '<none>'}"


def _summarize_flows(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    names: list[str] = []
    if isinstance(records, list):
        for item in records[:25]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("DeveloperName", "")).strip()
            if name:
                names.append(name)
    return f"flows={len(records) if isinstance(records, list) else 0}; sample={','.join(names)}"

def _build_flows_context(result: dict[str, Any]) -> str:
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        return "flows: unavailable"
    names: list[str] = []
    for item in records[:50]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("DeveloperName", "")).strip()
        version_obj = item.get("ActiveVersion")
        version = ""
        if isinstance(version_obj, dict):
            version = str(version_obj.get("VersionNumber", "")).strip()
        if name:
            names.append(f"{name}(v{version or '?'})")
    return f"flows:\n- count: {len(records)}\n- active_versions: {', '.join(names) or '<none>'}"


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
        lines.append("Facts / Rules:")
        for item in facts:
            kind = str(item.get("kind", "fact")).strip()
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            statement = str(item.get("statement", "")).strip()
            if statement:
                lines.append(f"- [{kind}] {title}: {statement}")
    if hypotheses:
        lines.append("Hypotheses:")
        for item in hypotheses:
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            statement = str(item.get("statement", "")).strip()
            if statement:
                lines.append(f"- {title}: {statement}")
    if questions:
        lines.append("Questions:")
        for item in questions:
            title = str(item.get("title", "Untitled")).strip() or "Untitled"
            question = str(item.get("question", "")).strip()
            if question:
                lines.append(f"- {title}: {question}")
    if len(lines) == 1:
        lines.append("- No facts, hypotheses, or questions were persisted.")
    return "\n".join(lines)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    normalized = _normalize_json_like_text(raw_text)
    attempts = [normalized]
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as exc:
            last_exc = exc
            candidate = _extract_first_json_object_text(attempt)
            if candidate is None:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as inner_exc:
                last_exc = inner_exc
                continue
    raise RuntimeError("Invalid ingestion JSON from model output.") from last_exc


def _normalize_json_like_text(raw_text: str) -> str:
    text = raw_text.strip().replace("\u00a0", " ")
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return text


def _truncate_for_log(text: str, max_chars: int = INGESTION_LOG_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... [truncated log output, omitted {omitted} chars]"


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


def _build_tool_calls_observability(events: list[dict[str, str]]) -> str:
    tool_events = [e for e in events if str(e.get("type", "")).startswith("tool_")]
    lines = ["Execution trace"]
    if not tool_events:
        lines.append("- No tool calls were executed.")
        return "```\n" + "\n".join(lines) + "\n```"
    for event in tool_events[:TOOL_TRACE_MAX_STEPS]:
        lines.append(
            f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}"
        )
    if len(tool_events) > TOOL_TRACE_MAX_STEPS:
        lines.append(f"- ... and {len(tool_events) - TOOL_TRACE_MAX_STEPS} more tool call(s)")
    return "```\n" + "\n".join(lines) + "\n```"


def _ingestion_execution_plan() -> list[str]:
    return [
        "Classify ingestion intent and set metadata-first scope.",
        "Run Salesforce discovery tools for validation rules, describes, naming, and automation metadata.",
        "Extract structured knowledge per stage with strict JSON.",
        "Persist facts/rules, hypotheses, and questions to database.",
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

