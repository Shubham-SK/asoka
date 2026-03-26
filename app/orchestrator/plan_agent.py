from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any
from typing import Callable

from app.config import Settings
from app.db.enums import KnowledgeKind, PlanStatus
from app.db.repository import (
    create_execution_plan,
    increment_knowledge_usage_counts,
    list_knowledge_for_retrieval,
    list_plan_summaries,
    list_pending_plan_summaries,
    set_execution_plan_status,
)
from app.db.session import SessionLocal
from app.llm.client import get_claude_client
from app.orchestrator.plan_backend import execute_approved_plan
from app.salesforce.client import get_salesforce_client

logger = logging.getLogger(__name__)

PLAN_AGENT_PROMPT = """
You are a workflow orchestrator for write-plan management in a Salesforce assistant.

Return ONLY valid JSON in one of these forms:
{"action":"tool","tool":"create_plan","input":{"summary":"...","operations":[{"id":"op1","op":"sobject_update","object":"Opportunity","record_id":"006...","fields":{"Amount":50000}}],"assumptions":[{"type":"...","value":"..."}]},"reason":"..."}
{"action":"tool","tool":"resolve_record","input":{"object":"Account","field":"Name","value":"Acme"},"reason":"..."}
{"action":"tool","tool":"describe_fields","input":{"object":"Opportunity","search":"ARR|amount","writeable_only":true,"limit":30},"reason":"..."}
{"action":"tool","tool":"list_plans","input":{"statuses":["pending_approval|approved|denied|draft|executed|failed"],"requester":"self|any|U123","limit":25},"reason":"..."}
{"action":"tool","tool":"list_pending_plans","input":{"limit":25},"reason":"..."}
{"action":"tool","tool":"approve_plan","input":{"plan_id":"<uuid>","reason":"..."},"reason":"..."}
{"action":"tool","tool":"reject_plan","input":{"plan_id":"<uuid>","reason":"..."},"reason":"..."}
{"action":"tool","tool":"role_scope","input":{},"reason":"..."}
{"action":"final","answer":"..."}

Rules:
- Never output non-JSON text.
- Use tools to inspect pending plans before approving/rejecting if the plan_id is unclear.
- For non-coworkers, never approve/reject plans; explain limitation in final answer.
- For non-coworkers, only query plans for requester=self.
- Keep final answers concise for Slack mrkdwn.
- Do not use markdown tables; use bullet lists or short paragraphs.
- After create_plan, include exact token `Plan ID: \`<uuid>\`` in final answer.
- If create_plan result status is pending_approval, include exact token `Status: \`pending_approval\``.
- After create_plan, include a short "Plan" section in the final answer with summary and operations.
- If approve/reject tools return `needs_clarification=true`, ask the user to pick one plan ID explicitly.
- For status questions like "denied?", "approved?", or "all plans", use `list_plans` (not `list_pending_plans`).
- For create_plan, prefer deterministic operations using supported ops:
  - sobject_create (object, fields)
  - sobject_update (object, record_id, fields)
  - sobject_upsert (object, external_id_field, external_id, fields)
  - sobject_delete (object, record_id)
- Prefer known object/field API names, but produce best-effort plans and let execution retries self-correct on API errors.
- Before choosing field API names for writes, call `describe_fields` for the target object when there is any ambiguity.
- For Opportunity ARR requests, prefer the matching custom ARR field discovered from `describe_fields`
  (for example `ARR_Expansion__c` for expansion ARR) instead of defaulting to `Amount`.
- For delete/update, include `record_id` when known; otherwise include `lookup` and let execution resolve it.
- Do not retry `resolve_record` by inventing alternate spellings/variants. Use `resolve_record` output candidates and ask user to confirm.
- For user deactivation/offboarding requests, include ownership transfer operations by default for:
  - Opportunity (OwnerId)
  - Lead (OwnerId)
  - Case (OwnerId)
- If any object has no matching open records, include the operation as a documented no-op assumption.
- You may include lookup hints when record id is unknown:
  - {"op":"sobject_delete","object":"Account","lookup":{"field":"Name","value":"Acme Inc"}}
  - {"op":"sobject_update","object":"Opportunity","lookup":{"field":"Name","value":"Big Renewal"},"fields":{...}}
"""

SF_ID_RE = re.compile(r"^[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?$")
SF_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def run_plan_agent(
    settings: Settings,
    user_text: str,
    workspace_id: str,
    requester_slack_user_id: str,
    is_coworker: bool,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    conversation_window: str = "",
    max_steps: int = 10,
    notify_pending_plan_callback: Callable[[str, str, str, str], None] | None = None,
    notify_plan_status_callback: Callable[[str, str, str, str, str, str], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    client = get_claude_client()
    events: list[dict[str, str]] = []
    created_plan_operations: list[dict[str, Any]] | None = None
    transcript: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "Execution context:\n"
                f"is_coworker={str(is_coworker).lower()}\n"
                f"workspace_id={workspace_id}\n"
                f"requester_slack_user_id={requester_slack_user_id}\n"
                f"recent_conversation_window={conversation_window or '<none>'}\n"
            ),
        },
        {
            "role": "user",
            "content": f"User request:\n{user_text}\n\nDecide first action.",
        },
    ]

    for _ in range(max_steps):
        action = _next_action(client=client, model=settings.llm_model, transcript=transcript)
        action_type = str(action.get("action", "")).strip()
        if action_type == "final":
            answer = str(action.get("answer", "")).strip()
            if answer:
                answer = _append_operations_outline(answer, created_plan_operations)
                return _build_observability_blob(events, parsed_intent, parsed_intent_reason) + "\n\n" + answer
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\n"
                + "I could not produce a final plan response."
            )

        if action_type != "tool":
            return (
                _build_observability_blob(events, parsed_intent, parsed_intent_reason)
                + "\n\n"
                + "I could not process that plan workflow request."
            )

        tool = str(action.get("tool", "")).strip()
        payload = action.get("input", {})
        action_reason = str(action.get("reason", "")).strip() or "no reason provided"
        if not isinstance(payload, dict):
            payload = {}

        events.append(
            {
                "step": str(len(events) + 1),
                "type": "tool",
                "status": "started",
                "reason": action_reason,
                "input": f"{tool}({json.dumps(payload, ensure_ascii=True)})",
                "output": "",
            }
        )
        _emit_progress_update(
            progress_callback=progress_callback,
            events=events,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
        )
        try:
            result = _run_tool(
                settings=settings,
                tool=tool,
                payload=payload,
                workspace_id=workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                is_coworker=is_coworker,
                user_text=user_text,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
                notify_pending_plan_callback=notify_pending_plan_callback,
                notify_plan_status_callback=notify_plan_status_callback,
            )
            events[-1]["status"] = "success"
            events[-1]["output"] = _truncate_text(json.dumps(result, ensure_ascii=True), 220)
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            if tool == "create_plan":
                plan_data = result.get("plan") if isinstance(result, dict) else None
                if isinstance(plan_data, dict) and isinstance(plan_data.get("operations"), list):
                    created_plan_operations = plan_data.get("operations")
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append(
                {
                    "role": "user",
                    "content": f"Tool result ({tool}): {json.dumps(result, ensure_ascii=True)}",
                }
            )
        except Exception as exc:
            logger.info("Plan agent tool failed: %s", exc)
            events[-1]["status"] = "error"
            events[-1]["output"] = _truncate_text(f"{type(exc).__name__}: {exc}", 220)
            _emit_progress_update(
                progress_callback=progress_callback,
                events=events,
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
            )
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append(
                {
                    "role": "user",
                    "content": f"Tool error ({tool}): {type(exc).__name__}: {exc}",
                }
            )

    return (
        _build_observability_blob(events, parsed_intent, parsed_intent_reason)
        + "\n\n"
        + "I reached the plan orchestration step limit. Please retry with a more specific request."
    )


def _run_tool(
    settings: Settings,
    tool: str,
    payload: dict[str, Any],
    workspace_id: str,
    requester_slack_user_id: str,
    is_coworker: bool,
    user_text: str,
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
    notify_pending_plan_callback: Callable[[str, str, str, str], None] | None = None,
    notify_plan_status_callback: Callable[[str, str, str, str, str, str], None] | None = None,
) -> dict[str, Any]:
    if tool == "resolve_record":
        object_name = str(payload.get("object", "")).strip()
        field_name = str(payload.get("field", "Name")).strip() or "Name"
        value = str(payload.get("value", "")).strip()
        if not object_name or not value:
            return {
                "error": "resolve_record requires non-empty `object` and `value`.",
                "needs_clarification": True,
            }
        return _resolve_record(
            workspace_id=workspace_id,
            requester_slack_user_id=requester_slack_user_id,
            object_name=object_name,
            field_name=field_name,
            field_value=value,
        )

    if tool == "describe_fields":
        object_name = str(payload.get("object", "")).strip()
        if not object_name:
            return {"error": "describe_fields requires non-empty `object`."}
        if not _is_safe_symbol(object_name):
            return {"error": f"Invalid object API name `{object_name}`."}
        writeable_only = bool(payload.get("writeable_only", False))
        limit = payload.get("limit", 50)
        if not isinstance(limit, int):
            limit = 50
        limit = max(1, min(limit, 200))
        search_text = str(payload.get("search", "")).strip().lower()
        search_terms = [term for term in re.split(r"\s+|\|", search_text) if term] if search_text else []

        sf = get_salesforce_client(slack_user_id=requester_slack_user_id, workspace_id=workspace_id)
        try:
            describe = sf.__getattr__(object_name).describe()
        except Exception as exc:
            return {
                "error": (
                    f"Could not describe object `{object_name}`: "
                    f"{type(exc).__name__}: {exc}"
                )
            }
        raw_fields = describe.get("fields", []) if isinstance(describe, dict) else []
        output_fields: list[dict[str, Any]] = []
        if isinstance(raw_fields, list):
            for item in raw_fields:
                if not isinstance(item, dict):
                    continue
                api_name = str(item.get("name", "")).strip()
                if not api_name:
                    continue
                label = str(item.get("label", "")).strip()
                is_createable = bool(item.get("createable", False))
                is_updateable = bool(item.get("updateable", False))
                if writeable_only and not (is_createable or is_updateable):
                    continue
                if search_terms:
                    searchable = f"{api_name} {label}".lower()
                    if not all(term in searchable for term in search_terms):
                        continue
                output_fields.append(
                    {
                        "name": api_name,
                        "label": label,
                        "type": str(item.get("type", "")).strip(),
                        "custom": bool(item.get("custom", False)),
                        "createable": is_createable,
                        "updateable": is_updateable,
                    }
                )
        output_fields = output_fields[:limit]
        return {
            "object": object_name,
            "count": len(output_fields),
            "fields": output_fields,
        }

    if tool == "role_scope":
        if is_coworker:
            return {
                "role": "coworker",
                "allowed": ["read_request", "write_request", "approve_or_reject_plans"],
            }
        return {
            "role": "end_user",
            "allowed": ["read_request", "write_request_plan_creation_only"],
        }

    if tool == "create_plan":
        summary = str(payload.get("summary", "")).strip() or user_text.strip()[:500]
        operations = _coerce_operations(payload.get("operations"))
        assumptions = payload.get("assumptions")
        if operations is None or not operations:
            return {
                "needs_clarification": True,
                "message": (
                    "Plan creation requires a strict JSON array in `operations` with executable "
                    "Salesforce operations."
                ),
            }
        operations, schema_issues = _sanitize_and_validate_operations_schema(operations)
        if schema_issues:
            return {
                "needs_clarification": True,
                "message": "Plan operations JSON is invalid.",
                "issues": schema_issues,
            }
        if not isinstance(assumptions, list):
            assumptions = []
        operations, resolution_issues = _hydrate_plan_operations(
            operations=operations,
            workspace_id=workspace_id,
            requester_slack_user_id=requester_slack_user_id,
        )
        assumptions.append({"type": "intent", "value": "write_request"})
        if resolution_issues:
            assumptions.append(
                {
                    "type": "resolution_warnings",
                    "value": resolution_issues[:5],
                }
            )
        if parsed_intent:
            assumptions.append({"type": "intent_parse", "value": parsed_intent})
        if parsed_intent_reason:
            assumptions.append({"type": "intent_reason", "value": parsed_intent_reason})

        with SessionLocal() as db:
            # All write plans require explicit review/approval, including coworker-created plans.
            status = PlanStatus.pending_approval
            plan = create_execution_plan(
                db=db,
                workspace_id=workspace_id,
                requester_slack_user_id=requester_slack_user_id,
                summary=summary,
                operations=operations,
                assumptions=assumptions,
                safety_checks=[],
                status=status,
            )
            plan_id = plan.id
            db.commit()
        if (
            status == PlanStatus.pending_approval
            and not is_coworker
            and notify_pending_plan_callback is not None
        ):
            try:
                notify_pending_plan_callback(
                    plan_id,
                    workspace_id,
                    requester_slack_user_id,
                    summary,
                )
            except Exception as exc:
                logger.info("Could not notify coworker for plan=%s: %s", plan_id, exc)
        return {
            "plan_id": plan_id,
            "status": status.value,
            "plan": {
                "summary": summary,
                "operations": operations,
                "assumptions": assumptions,
            },
        }

    if tool == "list_pending_plans":
        limit = payload.get("limit", 25)
        if not isinstance(limit, int):
            limit = 25
        limit = max(1, min(limit, 50))
        requester = None if is_coworker else requester_slack_user_id
        with SessionLocal() as db:
            plans = list_pending_plan_summaries(
                db=db,
                workspace_id=workspace_id,
                requester_slack_user_id=requester,
                limit=limit,
            )
        return {"pending_plans": plans}

    if tool == "list_plans":
        limit = payload.get("limit", 25)
        if not isinstance(limit, int):
            limit = 25
        limit = max(1, min(limit, 50))

        requester_raw = str(payload.get("requester", "self")).strip().lower()
        requester: str | None
        if not is_coworker:
            requester = requester_slack_user_id
        elif requester_raw in {"self", ""}:
            requester = requester_slack_user_id
        elif requester_raw == "any":
            requester = None
        else:
            requester = str(payload.get("requester", "")).strip() or requester_slack_user_id

        statuses_raw = payload.get("statuses", [])
        statuses: list[PlanStatus] = []
        if isinstance(statuses_raw, list):
            for value in statuses_raw:
                text = str(value).strip().lower()
                for enum_val in PlanStatus:
                    if enum_val.value == text:
                        statuses.append(enum_val)
                        break

        with SessionLocal() as db:
            plans = list_plan_summaries(
                db=db,
                workspace_id=workspace_id,
                statuses=statuses or None,
                requester_slack_user_id=requester,
                limit=limit,
            )
        return {
            "plans": plans,
            "filters": {
                "statuses": [s.value for s in statuses],
                "requester": requester or "any",
                "limit": limit,
            },
        }

    if tool == "approve_plan":
        if not is_coworker:
            return {"error": "Only coworker can approve plans."}
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            with SessionLocal() as db:
                plans = list_pending_plan_summaries(db=db, workspace_id=workspace_id, limit=10)
            return {
                "needs_clarification": True,
                "message": "Please specify which plan to approve.",
                "pending_plans": plans,
            }
        reason = str(payload.get("reason", "")).strip()
        with SessionLocal() as db:
            try:
                plan = set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.approved,
                    reason=reason,
                    actor_slack_user_id=requester_slack_user_id,
                    allowed_from_statuses=[PlanStatus.pending_approval],
                )
            except ValueError as exc:
                return {"error": str(exc)}
            if plan is None:
                return {"error": f"No plan found: {plan_id}"}
            target_requester = plan.requester_slack_user_id
            db.commit()
        if notify_plan_status_callback is not None:
            try:
                notify_plan_status_callback(
                    plan_id,
                    workspace_id,
                    target_requester,
                    PlanStatus.approved.value,
                    reason,
                    requester_slack_user_id,
                )
            except Exception as exc:
                logger.info("Could not notify requester for approved plan=%s: %s", plan_id, exc)
        if settings.plan_execute_on_approve:
            execution = execute_approved_plan(
                settings=settings,
                workspace_id=workspace_id,
                plan_id=plan_id,
            )
            if notify_plan_status_callback is not None and execution.status in {"executed", "failed"}:
                try:
                    notify_plan_status_callback(
                        plan_id,
                        workspace_id,
                        target_requester,
                        execution.status,
                        _with_observability(execution.message, execution.observability),
                        "plan_backend",
                    )
                except Exception as exc:
                    logger.info("Could not notify requester for execution status plan=%s: %s", plan_id, exc)
        return {"plan_id": plan_id, "status": PlanStatus.approved.value}

    if tool == "reject_plan":
        if not is_coworker:
            return {"error": "Only coworker can reject plans."}
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            with SessionLocal() as db:
                plans = list_pending_plan_summaries(db=db, workspace_id=workspace_id, limit=10)
            return {
                "needs_clarification": True,
                "message": "Please specify which plan to reject.",
                "pending_plans": plans,
            }
        reason = str(payload.get("reason", "")).strip()
        with SessionLocal() as db:
            try:
                plan = set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.denied,
                    reason=reason,
                    actor_slack_user_id=requester_slack_user_id,
                    allowed_from_statuses=[PlanStatus.pending_approval],
                )
            except ValueError as exc:
                return {"error": str(exc)}
            if plan is None:
                return {"error": f"No plan found: {plan_id}"}
            target_requester = plan.requester_slack_user_id
            db.commit()
        if notify_plan_status_callback is not None:
            try:
                notify_plan_status_callback(
                    plan_id,
                    workspace_id,
                    target_requester,
                    PlanStatus.denied.value,
                    reason,
                    requester_slack_user_id,
                )
            except Exception as exc:
                logger.info("Could not notify requester for denied plan=%s: %s", plan_id, exc)
        return {"plan_id": plan_id, "status": PlanStatus.denied.value, "reason": reason}

    return {"error": f"Unknown tool: {tool}"}


def _hydrate_plan_operations(
    operations: list[Any],
    workspace_id: str,
    requester_slack_user_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hydrated: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for idx, raw_op in enumerate(operations):
        if not isinstance(raw_op, dict):
            continue
        op = dict(raw_op)
        op_type = str(op.get("op", "")).strip().lower()
        object_name = str(op.get("object", "")).strip()
        if op_type not in {"sobject_update", "sobject_delete"}:
            hydrated.append(op)
            continue
        record_id = str(op.get("record_id", "")).strip()
        if record_id and _is_salesforce_id(record_id):
            hydrated.append(op)
            continue

        lookup = op.get("lookup", {})
        if not isinstance(lookup, dict):
            lookup = {}
        field_name = str(lookup.get("field", "Name")).strip() or "Name"
        field_value = str(lookup.get("value", "")).strip()
        if not field_value:
            issues.append(
                {
                    "operation_index": idx + 1,
                    "operation": op_type,
                    "object": object_name,
                    "reason": "Missing record_id and missing lookup.value.",
                }
            )
            hydrated.append(op)
            continue
        resolution = _resolve_record(
            workspace_id=workspace_id,
            requester_slack_user_id=requester_slack_user_id,
            object_name=object_name,
            field_name=field_name,
            field_value=field_value,
        )
        if "record_id" in resolution:
            op["record_id"] = str(resolution["record_id"]).strip()
            hydrated.append(op)
            continue
        issues.append(
            {
                "operation_index": idx + 1,
                "operation": op_type,
                "object": object_name,
                "lookup": {"field": field_name, "value": field_value},
                "reason": str(resolution.get("error", "record resolution failed")),
                "matches": resolution.get("matches", []),
            }
        )
        hydrated.append(op)
    return hydrated, issues


def _coerce_operations(raw: Any) -> list[Any] | None:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _sanitize_and_validate_operations_schema(
    operations: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sanitized: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for idx, raw_op in enumerate(operations):
        op_idx = idx + 1
        if not isinstance(raw_op, dict):
            issues.append({"operation_index": op_idx, "reason": "Operation must be an object."})
            continue
        op_type = str(raw_op.get("op", "")).strip().lower()
        object_name = str(raw_op.get("object", "")).strip()
        if op_type not in {"sobject_create", "sobject_update", "sobject_upsert", "sobject_delete"}:
            issues.append(
                {
                    "operation_index": op_idx,
                    "reason": f"Unsupported op `{op_type}`.",
                }
            )
            continue
        if not object_name:
            issues.append({"operation_index": op_idx, "reason": "Missing `object`."})
            continue

        clean_op: dict[str, Any] = {"op": op_type, "object": object_name}
        if op_type == "sobject_create":
            fields = raw_op.get("fields")
            if not isinstance(fields, dict) or not fields:
                issues.append(
                    {"operation_index": op_idx, "reason": "create requires non-empty `fields` object."}
                )
                continue
            clean_op["fields"] = fields
            sanitized.append(clean_op)
            continue

        if op_type == "sobject_update":
            fields = raw_op.get("fields")
            if not isinstance(fields, dict) or not fields:
                issues.append(
                    {"operation_index": op_idx, "reason": "update requires non-empty `fields` object."}
                )
                continue
            record_id = str(raw_op.get("record_id", "")).strip()
            lookup = raw_op.get("lookup")
            if record_id:
                clean_op["record_id"] = record_id
            elif isinstance(lookup, dict) and str(lookup.get("value", "")).strip():
                clean_op["lookup"] = {
                    "field": str(lookup.get("field", "Name")).strip() or "Name",
                    "value": str(lookup.get("value", "")).strip(),
                }
            else:
                issues.append(
                    {
                        "operation_index": op_idx,
                        "reason": "update requires `record_id` or `lookup.{field,value}`.",
                    }
                )
                continue
            clean_op["fields"] = fields
            sanitized.append(clean_op)
            continue

        if op_type == "sobject_upsert":
            external_id_field = str(raw_op.get("external_id_field", "")).strip()
            external_id = str(raw_op.get("external_id", "")).strip()
            fields = raw_op.get("fields")
            if not external_id_field or not external_id:
                issues.append(
                    {
                        "operation_index": op_idx,
                        "reason": "upsert requires `external_id_field` and `external_id`.",
                    }
                )
                continue
            if not isinstance(fields, dict) or not fields:
                issues.append(
                    {"operation_index": op_idx, "reason": "upsert requires non-empty `fields` object."}
                )
                continue
            clean_op["external_id_field"] = external_id_field
            clean_op["external_id"] = external_id
            clean_op["fields"] = fields
            sanitized.append(clean_op)
            continue

        record_id = str(raw_op.get("record_id", "")).strip()
        lookup = raw_op.get("lookup")
        if record_id:
            clean_op["record_id"] = record_id
            sanitized.append(clean_op)
            continue
        if isinstance(lookup, dict) and str(lookup.get("value", "")).strip():
            clean_op["lookup"] = {
                "field": str(lookup.get("field", "Name")).strip() or "Name",
                "value": str(lookup.get("value", "")).strip(),
            }
            sanitized.append(clean_op)
            continue
        issues.append(
            {
                "operation_index": op_idx,
                "reason": "delete requires `record_id` or `lookup.{field,value}`.",
            }
        )
    return sanitized, issues


def _resolve_record(
    workspace_id: str,
    requester_slack_user_id: str,
    object_name: str,
    field_name: str,
    field_value: str,
) -> dict[str, Any]:
    object_name = object_name.strip()
    field_name = field_name.strip() or "Name"
    field_value = field_value.strip()
    if not _is_safe_symbol(object_name):
        return {"error": f"Invalid object API name `{object_name}`."}
    if not _is_safe_symbol(field_name):
        return {"error": f"Invalid field API name `{field_name}`."}
    sf = get_salesforce_client(slack_user_id=requester_slack_user_id, workspace_id=workspace_id)
    select_fields: list[str] = []
    for candidate in ["Id", "Name", field_name]:
        if candidate not in select_fields:
            select_fields.append(candidate)
    soql = (
        f"SELECT {', '.join(select_fields)} FROM {object_name} "
        f"WHERE {field_name} = '{_soql_escape(field_value)}' LIMIT 2"
    )
    result = sf.query(soql)
    records = result.get("records", []) if isinstance(result, dict) else []
    ids = [str(item.get("Id", "")).strip() for item in records if isinstance(item, dict)]
    ids = [rid for rid in ids if rid]
    if not ids:
        if field_name.lower() == "name":
            return _resolve_record_name_suggestions(
                sf=sf,
                object_name=object_name,
                field_value=field_value,
            )
        return {
            "error": f"No {object_name} record found with {field_name}='{field_value}'.",
            "matches": [],
        }
    if len(ids) > 1:
        return {
            "error": f"Ambiguous match for {object_name} {field_name}='{field_value}'.",
            "matches": [
                {
                    "record_id": str(item.get("Id", "")).strip(),
                    "name": str(item.get("Name", "")).strip(),
                }
                for item in records
                if isinstance(item, dict) and str(item.get("Id", "")).strip()
            ],
            "needs_clarification": True,
        }
    return {
        "object": object_name,
        "field": field_name,
        "value": field_value,
        "record_id": ids[0],
    }


def _resolve_record_name_suggestions(
    sf: Any,
    object_name: str,
    field_value: str,
) -> dict[str, Any]:
    records, truncated = _fetch_all_name_candidates(sf=sf, object_name=object_name)
    if not records:
        return {
            "error": f"No {object_name} records with non-empty Name were found to compare against.",
            "matches": [],
            "needs_clarification": True,
        }

    scored: list[tuple[float, dict[str, Any]]] = []
    target = _normalize_name_for_match(field_value)
    for item in records:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("Id", "")).strip()
        name = str(item.get("Name", "")).strip()
        if not rid or not name:
            continue
        score = difflib.SequenceMatcher(None, target, _normalize_name_for_match(name)).ratio()
        scored.append((score, {"record_id": rid, "name": name, "similarity": round(score, 3)}))
    if not scored:
        return {
            "error": f"No comparable {object_name} Name values were found.",
            "matches": [],
            "needs_clarification": True,
        }

    scored.sort(key=lambda x: x[0], reverse=True)

    top_matches = [item for _, item in scored[:8]]
    note = (
        "Scanned many records and found close candidates."
        if not truncated
        else "Scanned up to safety cap; results may be incomplete."
    )
    return {
        "error": (
            f"No exact match for {object_name} Name='{field_value}'. "
            "Please choose one of these close matches."
        ),
        "matches": top_matches,
        "note": note,
        "needs_clarification": True,
    }


def _fetch_all_name_candidates(sf: Any, object_name: str) -> tuple[list[dict[str, Any]], bool]:
    """
    Fetch Name candidates without inventing query variants.
    Uses full pagination with a safety cap to avoid runaway scans.
    """
    safety_cap = 20000
    soql = f"SELECT Id, Name FROM {object_name} WHERE Name != null ORDER BY Name"
    result = sf.query(soql)
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        records = []

    done = bool(result.get("done", True)) if isinstance(result, dict) else True
    next_url = result.get("nextRecordsUrl") if isinstance(result, dict) else None
    truncated = False

    while not done and next_url:
        if len(records) >= safety_cap:
            truncated = True
            break
        next_result = sf.query_more(next_url, True)
        batch = next_result.get("records", []) if isinstance(next_result, dict) else []
        if isinstance(batch, list):
            records.extend(batch)
        done = bool(next_result.get("done", True)) if isinstance(next_result, dict) else True
        next_url = next_result.get("nextRecordsUrl") if isinstance(next_result, dict) else None

    if len(records) > safety_cap:
        records = records[:safety_cap]
        truncated = True
    return records, truncated


def _normalize_name_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _next_action(client: Any, model: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0,
        system=PLAN_AGENT_PROMPT,
        messages=transcript,
    )
    text_parts: list[str] = []
    for part in response.content:
        if getattr(part, "type", "") == "text":
            text_parts.append(part.text)
    raw = "\n".join(text_parts).strip()
    try:
        return _extract_json_object(raw)
    except RuntimeError as exc:
        logger.warning(
            "Plan agent action JSON parse failed; attempting repair. raw_len=%s raw_preview=%s error=%s",
            len(raw),
            _truncate_text(raw.replace("\n", " "), 280),
            exc,
        )
        repaired = _repair_action_json(client=client, model=model, raw_text=raw)
        if repaired is not None:
            logger.info("Plan agent action JSON repair succeeded.")
            return repaired
        if raw:
            # Avoid leaking raw tool payloads in user-visible output.
            lowered = raw.lower()
            if '"action"' in lowered and '"tool"' in lowered:
                logger.warning(
                    "Plan agent emitted tool-like text after repair failure; returning safe retry message. "
                    "raw_preview=%s",
                    _truncate_text(raw.replace("\n", " "), 280),
                )
                return {
                    "action": "final",
                    "answer": (
                        "I hit a temporary formatting issue while planning. "
                        "Please retry your request."
                    ),
                }
            # If the model emits plain text instead of JSON, treat it as final.
            logger.info("Plan agent returned plain-text final fallback after JSON repair failure.")
            return {"action": "final", "answer": raw}
        raise


def _repair_action_json(client: Any, model: str, raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    try:
        response = client.messages.create(
            model=model,
            max_tokens=350,
            temperature=0,
            system=(
                "Convert the input into ONE strict JSON object only.\n"
                "Allowed schemas:\n"
                '{"action":"tool","tool":"...","input":{...},"reason":"..."}\n'
                '{"action":"final","answer":"..."}\n'
                "If uncertain, return final answer schema."
            ),
            messages=[{"role": "user", "content": text}],
        )
        repaired_parts: list[str] = []
        for part in response.content:
            if getattr(part, "type", "") == "text":
                repaired_parts.append(part.text)
        repaired_raw = "\n".join(repaired_parts).strip()
        if not repaired_raw:
            return None
        parsed = _extract_json_object(repaired_raw)
        action_type = str(parsed.get("action", "")).strip()
        if action_type == "tool":
            tool = str(parsed.get("tool", "")).strip()
            payload = parsed.get("input", {})
            if tool and isinstance(payload, dict):
                return parsed
            return None
        if action_type == "final":
            answer = str(parsed.get("answer", "")).strip()
            if answer:
                return {"action": "final", "answer": answer}
        return None
    except Exception as exc:
        logger.warning(
            "Plan agent JSON repair attempt failed. raw_preview=%s error=%s",
            _truncate_text(text.replace("\n", " "), 280),
            exc,
        )
        return None


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


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _with_observability(message: str, observability: str) -> str:
    if observability.strip():
        return f"{message}\n\n{observability}"
    return message


def _emit_progress_update(
    progress_callback: Callable[[str], None] | None,
    events: list[dict[str, str]],
    parsed_intent: str = "",
    parsed_intent_reason: str = "",
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(_build_observability_blob(events, parsed_intent, parsed_intent_reason))
    except Exception as exc:
        logger.info("Could not emit plan progress update: %s", exc)


def _append_operations_outline(
    answer: str,
    operations: list[dict[str, Any]] | None,
) -> str:
    if not operations:
        return answer
    lines = ["Planned operations:"]
    for idx, op in enumerate(operations[:10], start=1):
        op_type = str(op.get("op", "")).strip() or "unknown_op"
        object_name = str(op.get("object", "")).strip() or "UnknownObject"
        target = str(op.get("record_id", "")).strip()
        if not target:
            lookup = op.get("lookup")
            if isinstance(lookup, dict) and str(lookup.get("value", "")).strip():
                field = str(lookup.get("field", "Name")).strip() or "Name"
                value = str(lookup.get("value", "")).strip()
                target = f"{field}={value}"
        if not target and op_type == "sobject_upsert":
            ext_field = str(op.get("external_id_field", "")).strip()
            ext_id = str(op.get("external_id", "")).strip()
            if ext_field and ext_id:
                target = f"{ext_field}={ext_id}"
        suffix = f" ({target})" if target else ""
        lines.append(f"- {idx}. {op_type} {object_name}{suffix}")
    if len(operations) > 10:
        lines.append(f"- ... and {len(operations) - 10} more operation(s)")
    block = "\n".join(lines)
    if block in answer:
        return answer
    return f"{answer}\n\n{block}"


def _is_salesforce_id(value: str) -> bool:
    return bool(SF_ID_RE.fullmatch(value.strip()))


def _is_safe_symbol(value: str) -> bool:
    return bool(SF_SYMBOL_RE.fullmatch(value.strip()))


def _soql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _knowledge_precheck_create_plan(
    db: Any,
    workspace_id: str,
    operations: list[dict[str, Any]],
    user_text: str,
) -> dict[str, Any]:
    knowledge_items = list_knowledge_for_retrieval(
        db=db,
        workspace_id=workspace_id,
        kinds=[KnowledgeKind.rule, KnowledgeKind.fact, KnowledgeKind.question],
        min_confidence_rank=1,
        limit=120,
    )
    increment_knowledge_usage_counts(
        db=db,
        workspace_id=workspace_id,
        knowledge_item_ids=[item.id for item in knowledge_items],
    )
    required_fields = _infer_required_fields_from_knowledge(knowledge_items)
    tier_minimums, global_minimum = _infer_amount_thresholds_from_knowledge(knowledge_items)
    text_lower = user_text.lower()
    inferred_tier = _infer_tier_from_text(text_lower)
    missing_requirements: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for op in operations:
        op_type = str(op.get("op", "")).strip().lower()
        object_name = str(op.get("object", "")).strip()
        fields = op.get("fields", {})
        if not isinstance(fields, dict):
            fields = {}
        if op_type == "sobject_create":
            reqs = required_fields.get(object_name, set())
            if reqs:
                present = {str(k).strip().lower() for k in fields.keys()}
                missing = [name for name in reqs if name.lower() not in present]
                if missing:
                    missing_requirements.append(
                        {
                            "operation": op_type,
                            "object": object_name,
                            "missing_fields": missing,
                        }
                    )
        if op_type in {"sobject_create", "sobject_update"} and object_name.lower() == "opportunity":
            amount = _parse_numeric(fields.get("Amount"))
            if amount is None:
                continue
            threshold = global_minimum
            if inferred_tier and inferred_tier in tier_minimums:
                threshold = tier_minimums[inferred_tier]
            if threshold is not None and amount < threshold:
                warnings.append(
                    {
                        "operation": op_type,
                        "object": object_name,
                        "field": "Amount",
                        "provided": amount,
                        "minimum_expected": threshold,
                        "message": (
                            f"This amount may violate known policy (minimum {threshold:.0f}). "
                            "Are you sure you want to proceed?"
                        ),
                    }
                )

    clarification_questions: list[str] = []
    for item in knowledge_items:
        if item.kind != KnowledgeKind.question:
            continue
        question = str((item.content_json or {}).get("question", "")).strip()
        if question:
            clarification_questions.append(question)
        if len(clarification_questions) >= 3:
            break
    return {
        "missing_requirements": missing_requirements,
        "warnings": warnings,
        "clarification_questions": clarification_questions,
    }


def _infer_required_fields_from_knowledge(knowledge_items: list[Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for item in knowledge_items:
        if item.kind not in {KnowledgeKind.rule, KnowledgeKind.fact}:
            continue
        text = f"{item.title} {(item.content_json or {}).get('statement', '')}"
        lowered = text.lower()
        if "requires" not in lowered or "to be saved" not in lowered:
            continue
        object_name = _infer_object_name(text)
        if not object_name:
            continue
        match = re.search(r"requires\s+(.+?)\s+to be saved", text, flags=re.IGNORECASE)
        if not match:
            continue
        fields_blob = match.group(1)
        parts = re.split(r",| and ", fields_blob)
        names = {part.strip() for part in parts if part.strip()}
        if names:
            out.setdefault(object_name, set()).update(names)
    return out


def _infer_amount_thresholds_from_knowledge(
    knowledge_items: list[Any],
) -> tuple[dict[str, float], float | None]:
    by_tier: dict[str, float] = {}
    global_minimum: float | None = None
    for item in knowledge_items:
        if item.kind not in {KnowledgeKind.rule, KnowledgeKind.fact}:
            continue
        text = f"{item.title} {(item.content_json or {}).get('statement', '')}"
        for tier, amount_text in re.findall(
            r"(enterprise|mid-market|smb)[^$]{0,60}\$([0-9][0-9,]*)",
            text,
            flags=re.IGNORECASE,
        ):
            amount = _parse_numeric(amount_text)
            if amount is not None:
                by_tier[tier.lower()] = amount
        for amount_text in re.findall(r"at least\s*\$([0-9][0-9,]*)", text, flags=re.IGNORECASE):
            amount = _parse_numeric(amount_text)
            if amount is None:
                continue
            if global_minimum is None or amount > global_minimum:
                global_minimum = amount
    return by_tier, global_minimum


def _parse_numeric(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _infer_object_name(text: str) -> str:
    lowered = text.lower()
    if "opportunity" in lowered:
        return "Opportunity"
    if "account" in lowered:
        return "Account"
    if "lead" in lowered:
        return "Lead"
    if "case" in lowered or "cases" in lowered:
        return "Case"
    return ""


def _infer_tier_from_text(text: str) -> str:
    if "enterprise" in text:
        return "enterprise"
    if "mid-market" in text or "mid market" in text:
        return "mid-market"
    if "smb" in text:
        return "smb"
    return ""


def _validate_operations_against_salesforce_metadata(
    workspace_id: str,
    requester_slack_user_id: str,
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sf = get_salesforce_client(slack_user_id=requester_slack_user_id, workspace_id=workspace_id)
    describe_cache: dict[str, dict[str, Any]] = {}

    def _describe(object_name: str) -> dict[str, Any] | None:
        if object_name in describe_cache:
            return describe_cache[object_name]
        try:
            describe_cache[object_name] = sf.__getattr__(object_name).describe()
            return describe_cache[object_name]
        except Exception as exc:
            issues.append(
                {
                    "object": object_name,
                    "reason": f"Object describe failed: {type(exc).__name__}",
                }
            )
            describe_cache[object_name] = {}
            return None

    for idx, op in enumerate(operations):
        op_type = str(op.get("op", "")).strip().lower()
        object_name = str(op.get("object", "")).strip()
        if not object_name:
            continue
        describe = _describe(object_name)
        if not describe:
            issues.append(
                {
                    "operation_index": idx + 1,
                    "operation": op_type,
                    "object": object_name,
                    "reason": "Unknown or inaccessible Salesforce object.",
                }
            )
            continue
        fields = describe.get("fields", [])
        field_map: dict[str, dict[str, Any]] = {}
        if isinstance(fields, list):
            for item in fields:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name:
                    field_map[name] = item

        op_fields = op.get("fields", {})
        if isinstance(op_fields, dict):
            for field_name in op_fields.keys():
                normalized = str(field_name).strip()
                if normalized not in field_map:
                    issues.append(
                        {
                            "operation_index": idx + 1,
                            "operation": op_type,
                            "object": object_name,
                            "field": normalized,
                            "reason": "Field does not exist on object.",
                        }
                    )
                    continue
                field_meta = field_map.get(normalized, {})
                if op_type == "sobject_create" and not bool(field_meta.get("createable", True)):
                    issues.append(
                        {
                            "operation_index": idx + 1,
                            "operation": op_type,
                            "object": object_name,
                            "field": normalized,
                            "reason": "Field is not writeable on create for this user/API profile.",
                        }
                    )
                if op_type == "sobject_update" and not bool(field_meta.get("updateable", True)):
                    issues.append(
                        {
                            "operation_index": idx + 1,
                            "operation": op_type,
                            "object": object_name,
                            "field": normalized,
                            "reason": "Field is not writeable on update for this user/API profile.",
                        }
                    )
                if op_type == "sobject_upsert" and not (
                    bool(field_meta.get("createable", True)) or bool(field_meta.get("updateable", True))
                ):
                    issues.append(
                        {
                            "operation_index": idx + 1,
                            "operation": op_type,
                            "object": object_name,
                            "field": normalized,
                            "reason": "Field is not writeable for upsert for this user/API profile.",
                        }
                    )

        if op_type == "sobject_create":
            provided = {str(k).strip() for k in op_fields.keys()} if isinstance(op_fields, dict) else set()
            required: list[str] = []
            for f_name, meta in field_map.items():
                nillable = bool(meta.get("nillable", True))
                defaulted = bool(meta.get("defaultedOnCreate", False))
                auto_number = bool(meta.get("autoNumber", False))
                calculated = bool(meta.get("calculated", False))
                createable = bool(meta.get("createable", True))
                if createable and (not nillable) and (not defaulted) and (not auto_number) and (not calculated):
                    required.append(f_name)
            missing = [name for name in required if name not in provided]
            if missing:
                issues.append(
                    {
                        "operation_index": idx + 1,
                        "operation": op_type,
                        "object": object_name,
                        "missing_fields": missing[:20],
                        "reason": "Missing required create fields.",
                    }
                )

        if op_type == "sobject_upsert":
            ext_field = str(op.get("external_id_field", "")).strip()
            if ext_field and ext_field in field_map:
                if not bool(field_map[ext_field].get("externalId", False)):
                    issues.append(
                        {
                            "operation_index": idx + 1,
                            "operation": op_type,
                            "object": object_name,
                            "field": ext_field,
                            "reason": "Field is not marked as External ID.",
                        }
                    )
            elif ext_field:
                issues.append(
                    {
                        "operation_index": idx + 1,
                        "operation": op_type,
                        "object": object_name,
                        "field": ext_field,
                        "reason": "External ID field does not exist on object.",
                    }
                )

    return issues
