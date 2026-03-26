from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.db.enums import PlanStatus
from app.db.repository import get_execution_plan_for_workspace, set_execution_plan_status
from app.db.session import SessionLocal
from app.llm.client import get_claude_client
from app.salesforce.client import get_salesforce_client

logger = logging.getLogger(__name__)
SF_ID_RE = re.compile(r"^[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?$")
REF_TOKEN_RE = re.compile(r"^@ref:([A-Za-z0-9_\-]+)(?:\.(?:id|Id))?$")


@dataclass
class PlanExecutionResult:
    status: str
    message: str
    completed_ops: int = 0
    observability: str = ""


def execute_approved_plan(
    settings: Settings,
    workspace_id: str,
    plan_id: str,
) -> PlanExecutionResult:
    events: list[dict[str, str]] = []
    if settings.plan_backend.strip().lower() != "salesforce_api":
        return PlanExecutionResult(
            status="skipped",
            message=f"Execution skipped: PLAN_BACKEND is `{settings.plan_backend}`.",
            completed_ops=0,
            observability=_build_observability_blob(events),
        )

    with SessionLocal() as db:
        plan = get_execution_plan_for_workspace(db=db, workspace_id=workspace_id, plan_id=plan_id)
        if plan is None:
            events.append(
                {
                    "step": "1",
                    "type": "load_plan",
                    "status": "error",
                    "reason": f"No plan found for id {plan_id}",
                }
            )
            return PlanExecutionResult(
                status="error",
                message=f"No plan found: {plan_id}",
                observability=_build_observability_blob(events),
            )
        if plan.status != PlanStatus.approved:
            events.append(
                {
                    "step": "1",
                    "type": "status_check",
                    "status": "error",
                    "reason": f"Plan status must be approved, found {plan.status.value}",
                }
            )
            return PlanExecutionResult(
                status="error",
                message=f"Plan must be approved before execution (current={plan.status.value}).",
                observability=_build_observability_blob(events),
            )
        requester = plan.requester_slack_user_id
        operations = _normalized_operations(plan.operations_json)

    try:
        _validate_operations(operations)
    except ValueError as exc:
        events.append(
            {
                "step": "1",
                "type": "validate_operations",
                "status": "error",
                "reason": str(exc),
            }
        )
        with SessionLocal() as db:
            try:
                set_execution_plan_status(
                    db=db,
                    workspace_id=workspace_id,
                    plan_id=plan_id,
                    status=PlanStatus.failed,
                    reason=f"Invalid operations payload: {exc}",
                    actor_slack_user_id="plan_backend",
                    allowed_from_statuses=[PlanStatus.approved],
                )
                db.commit()
            except Exception as status_exc:
                logger.info("Could not persist failed status for invalid plan payload: %s", status_exc)
        return PlanExecutionResult(
            status="failed",
            message=f"Invalid operations payload: {exc}",
            observability=_build_observability_blob(events),
        )

    sf = get_salesforce_client(slack_user_id=requester, workspace_id=workspace_id)
    completed = 0
    ref_values: dict[str, str] = {}
    for idx, op in enumerate(operations):
        op_type = str(op.get("op", "")).strip().lower() or "unknown_op"
        step = str(idx + 1)
        alias = _operation_alias(op, idx)
        events.append(
            {
                "step": step,
                "type": op_type,
                "status": "started",
                "reason": (
                    f"Execute deterministic operation on "
                    f"{str(op.get('object', '')).strip() or 'unknown object'}"
                ),
            }
        )
        try:
            resolved_op = _resolve_operation_references(op=op, ref_values=ref_values, op_number=idx + 1)
            resolved_op = _resolve_lookup_record_id(sf=sf, op=resolved_op)
            result, retry_note = _execute_one_with_retry(sf=sf, op=resolved_op, settings=settings)
            completed += 1
            _register_operation_reference(
                alias=alias,
                op=resolved_op,
                op_result=result,
                ref_values=ref_values,
            )
            events[-1]["status"] = "success"
            if retry_note:
                events[-1]["reason"] = f"{events[-1]['reason']} ({retry_note})"
        except Exception as exc:
            events[-1]["status"] = "error"
            events[-1]["reason"] = f"Execution failed: {type(exc).__name__}: {exc}"
            reason = f"Execution failed at op {idx + 1}: {type(exc).__name__}: {exc}"
            with SessionLocal() as db:
                try:
                    set_execution_plan_status(
                        db=db,
                        workspace_id=workspace_id,
                        plan_id=plan_id,
                        status=PlanStatus.failed,
                        reason=reason,
                        actor_slack_user_id="plan_backend",
                        allowed_from_statuses=[PlanStatus.approved],
                    )
                    db.commit()
                except Exception as status_exc:
                    logger.info("Could not persist failed status for execution error: %s", status_exc)
            return PlanExecutionResult(
                status="failed",
                message=reason,
                completed_ops=completed,
                observability=_build_observability_blob(events),
            )

    with SessionLocal() as db:
        set_execution_plan_status(
            db=db,
            workspace_id=workspace_id,
            plan_id=plan_id,
            status=PlanStatus.executed,
            reason=f"Executed {completed} operation(s) successfully.",
            actor_slack_user_id="plan_backend",
            allowed_from_statuses=[PlanStatus.approved],
        )
        db.commit()
    return PlanExecutionResult(
        status="executed",
        message=f"Executed {completed} operation(s) successfully.",
        completed_ops=completed,
        observability=_build_observability_blob(events),
    )


def _normalized_operations(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        ops = raw.get("ops", [])
        if isinstance(ops, list):
            return [item for item in ops if isinstance(item, dict)]
    return []


def _validate_operations(operations: list[dict[str, Any]]) -> None:
    if not operations:
        raise ValueError("No operations to execute.")
    aliases: dict[str, int] = {}
    for idx, op in enumerate(operations):
        alias = _operation_alias(op, idx)
        if alias in aliases:
            raise ValueError(
                f"Duplicate operation id/alias `{alias}` for operations #{aliases[alias] + 1} and #{idx + 1}."
            )
        aliases[alias] = idx

    for idx, op in enumerate(operations):
        op_type = str(op.get("op", "")).strip().lower()
        object_name = str(op.get("object", "")).strip()
        if not op_type or not object_name:
            raise ValueError(f"Operation #{idx + 1} missing `op` or `object`.")
        if op_type == "sobject_create":
            if not isinstance(op.get("fields"), dict) or not op.get("fields"):
                raise ValueError(f"Operation #{idx + 1} create requires non-empty `fields`.")
            continue
        if op_type == "sobject_update":
            record_id = str(op.get("record_id", "")).strip()
            lookup = op.get("lookup")
            has_lookup = isinstance(lookup, dict) and str(lookup.get("value", "")).strip()
            if not record_id and not has_lookup:
                raise ValueError(f"Operation #{idx + 1} update requires `record_id` or `lookup`.")
            if record_id and not _is_salesforce_id_or_ref(record_id):
                raise ValueError(
                    f"Operation #{idx + 1} update requires a Salesforce record id "
                    f"(15/18 alphanumeric chars) or `@ref:<op_id>`, got `{record_id}`."
                )
            if record_id:
                _validate_ref_token(record_id=record_id, aliases=aliases, current_idx=idx, op_number=idx + 1)
            if not isinstance(op.get("fields"), dict) or not op.get("fields"):
                raise ValueError(f"Operation #{idx + 1} update requires non-empty `fields`.")
            _validate_field_refs(
                fields=op.get("fields", {}),
                aliases=aliases,
                current_idx=idx,
                op_number=idx + 1,
            )
            continue
        if op_type == "sobject_upsert":
            if not str(op.get("external_id_field", "")).strip():
                raise ValueError(f"Operation #{idx + 1} upsert requires `external_id_field`.")
            if not str(op.get("external_id", "")).strip():
                raise ValueError(f"Operation #{idx + 1} upsert requires `external_id`.")
            if not isinstance(op.get("fields"), dict) or not op.get("fields"):
                raise ValueError(f"Operation #{idx + 1} upsert requires non-empty `fields`.")
            _validate_field_refs(
                fields=op.get("fields", {}),
                aliases=aliases,
                current_idx=idx,
                op_number=idx + 1,
            )
            continue
        if op_type == "sobject_delete":
            record_id = str(op.get("record_id", "")).strip()
            lookup = op.get("lookup")
            has_lookup = isinstance(lookup, dict) and str(lookup.get("value", "")).strip()
            if not record_id and not has_lookup:
                raise ValueError(f"Operation #{idx + 1} delete requires `record_id` or `lookup`.")
            if record_id and not _is_salesforce_id_or_ref(record_id):
                raise ValueError(
                    f"Operation #{idx + 1} delete requires a Salesforce record id "
                    f"(15/18 alphanumeric chars) or `@ref:<op_id>`, got `{record_id}`."
                )
            if record_id:
                _validate_ref_token(record_id=record_id, aliases=aliases, current_idx=idx, op_number=idx + 1)
            continue
        raise ValueError(f"Operation #{idx + 1} uses unsupported op `{op_type}`.")


def _execute_one(sf: Any, op: dict[str, Any]) -> Any:
    op_type = str(op.get("op", "")).strip().lower()
    object_name = str(op.get("object", "")).strip()
    fields = op.get("fields", {})
    obj = sf.__getattr__(object_name)
    if op_type == "sobject_create":
        return obj.create(fields)
    if op_type == "sobject_update":
        record_ids = op.get("record_ids")
        if isinstance(record_ids, list):
            ids = [str(item).strip() for item in record_ids if str(item).strip()]
            if not ids:
                return {"success": True, "updated": 0}
            results = [obj.update(record_id, fields) for record_id in ids]
            return {"success": True, "updated": len(ids), "results": results}
        return obj.update(str(op["record_id"]).strip(), fields)
    if op_type == "sobject_upsert":
        external_id_path = f"{str(op['external_id_field']).strip()}/{str(op['external_id']).strip()}"
        return obj.upsert(external_id_path, fields)
    if op_type == "sobject_delete":
        return obj.delete(str(op["record_id"]).strip())
    raise ValueError(f"Unsupported op `{op_type}`.")


def _execute_one_with_retry(sf: Any, op: dict[str, Any], settings: Settings) -> tuple[Any, str]:
    notes: list[str] = []
    current_op = dict(op)
    current_op = _resolve_lookup_record_id(sf=sf, op=current_op)

    try:
        return _execute_one(sf=sf, op=current_op), ""
    except Exception as first_exc:
        repaired, note = _repair_operation_from_error(sf=sf, op=current_op, exc=first_exc)
        if repaired is not None:
            notes.append(note)
            current_op = _resolve_lookup_record_id(sf=sf, op=repaired)
            try:
                return _execute_one(sf=sf, op=current_op), "; ".join(notes)
            except Exception as second_exc:
                llm_repaired, llm_note = _repair_operation_with_model(
                    settings=settings,
                    sf=sf,
                    op=current_op,
                    exc=second_exc,
                )
                if llm_repaired is None:
                    raise second_exc
                notes.append(llm_note)
                current_op = _resolve_lookup_record_id(sf=sf, op=llm_repaired)
                return _execute_one(sf=sf, op=current_op), "; ".join(notes)

        llm_repaired, llm_note = _repair_operation_with_model(
            settings=settings,
            sf=sf,
            op=current_op,
            exc=first_exc,
        )
        if llm_repaired is None:
            raise
        notes.append(llm_note)
        current_op = _resolve_lookup_record_id(sf=sf, op=llm_repaired)
        return _execute_one(sf=sf, op=current_op), "; ".join(notes)


def _repair_operation_from_error(
    sf: Any,
    op: dict[str, Any],
    exc: Exception,
) -> tuple[dict[str, Any] | None, str]:
    op_type = str(op.get("op", "")).strip().lower()
    if op_type not in {"sobject_create", "sobject_update", "sobject_upsert"}:
        return None, ""
    fields = op.get("fields", {})
    if not isinstance(fields, dict) or not fields:
        return None, ""

    text = str(exc)
    field_match = re.search(r"No such column '([^']+)' on sobject of type ([A-Za-z0-9_]+)", text)
    object_name = str(op.get("object", "")).strip()
    if field_match:
        bad_field = field_match.group(1).strip()
        object_from_error = field_match.group(2).strip()
        if object_name and object_from_error and object_name.lower() != object_from_error.lower():
            return None, ""
        if bad_field not in fields:
            return None, ""

        try:
            desc = sf.__getattr__(object_name).describe()
        except Exception:
            return None, ""
        desc_fields = desc.get("fields", []) if isinstance(desc, dict) else []
        field_names = [
            str(item.get("name", "")).strip()
            for item in desc_fields
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ]
        if not field_names:
            return None, ""
        candidates = difflib.get_close_matches(bad_field, field_names, n=2, cutoff=0.75)
        if len(candidates) != 1:
            return None, ""
        replacement = candidates[0]

        repaired = dict(op)
        repaired_fields = dict(fields)
        repaired_fields[replacement] = repaired_fields.pop(bad_field)
        repaired["fields"] = repaired_fields
        return repaired, f"auto-corrected field `{bad_field}` -> `{replacement}` and retried"

    # Handle FLS / API write restrictions by dropping blocked fields and retrying once.
    blocked_match = re.search(r"Unable to create/update fields:\s*([^.]+)\.", text)
    if blocked_match:
        blocked_blob = blocked_match.group(1).strip()
        blocked_fields = [part.strip() for part in blocked_blob.split(",") if part.strip()]
        blocked_set = set(blocked_fields)
        present_blocked = [name for name in fields.keys() if str(name).strip() in blocked_set]
        if not present_blocked:
            return None, ""
        repaired = dict(op)
        repaired_fields = {k: v for k, v in fields.items() if str(k).strip() not in blocked_set}
        if not repaired_fields and op_type in {"sobject_create", "sobject_update", "sobject_upsert"}:
            return None, ""
        repaired["fields"] = repaired_fields
        return repaired, f"removed non-writeable fields `{', '.join(present_blocked)}` and retried"
    return None, ""


def _resolve_lookup_record_id(sf: Any, op: dict[str, Any]) -> dict[str, Any]:
    op_type = str(op.get("op", "")).strip().lower()
    if op_type not in {"sobject_update", "sobject_delete"}:
        return op
    if str(op.get("record_id", "")).strip():
        return op
    lookup = op.get("lookup")
    if not isinstance(lookup, dict):
        return op
    object_name = str(op.get("object", "")).strip()
    requested_field = str(lookup.get("field", "Name")).strip() or "Name"
    field_value = str(lookup.get("value", "")).strip()
    if not object_name or not field_value:
        return op
    available_fields = _describe_field_names(sf=sf, object_name=object_name)
    field_name = _resolve_lookup_field_name(
        object_name=object_name,
        requested_field=requested_field,
        available_fields=available_fields,
    )

    soql = f"SELECT Id FROM {object_name} WHERE {field_name} = '{_soql_escape(field_value)}' LIMIT 201"
    result = sf.query(soql)
    records = result.get("records", []) if isinstance(result, dict) else []
    if not isinstance(records, list):
        records = []
    ids = [str(item.get("Id", "")).strip() for item in records if isinstance(item, dict)]
    ids = [rid for rid in ids if rid]
    op_type = str(op.get("op", "")).strip().lower()
    if op_type == "sobject_update":
        resolved = dict(op)
        if len(ids) == 1:
            resolved["record_id"] = ids[0]
            return resolved
        resolved["record_ids"] = ids
        return resolved

    if len(ids) != 1:
        raise ValueError(
            f"Could not resolve unique `{object_name}` record for lookup {field_name}='{field_value}'."
        )
    resolved = dict(op)
    resolved["record_id"] = ids[0]
    return resolved


def _repair_operation_with_model(
    settings: Settings,
    sf: Any,
    op: dict[str, Any],
    exc: Exception,
) -> tuple[dict[str, Any] | None, str]:
    object_name = str(op.get("object", "")).strip()
    fields = op.get("fields", {})
    describe_fields: list[str] = []
    if object_name:
        try:
            desc = sf.__getattr__(object_name).describe()
            raw_fields = desc.get("fields", []) if isinstance(desc, dict) else []
            if isinstance(raw_fields, list):
                describe_fields = [
                    str(item.get("name", "")).strip()
                    for item in raw_fields
                    if isinstance(item, dict) and str(item.get("name", "")).strip()
                ][:200]
        except Exception:
            describe_fields = []

    prompt = (
        "Repair this Salesforce operation to fix the execution error.\n"
        "Return ONLY JSON object with the same schema as input operation.\n"
        "Rules:\n"
        "- Keep op and object unless error clearly indicates they are wrong.\n"
        "- If a field does not exist, replace it with the closest valid field from allowed_fields.\n"
        "- If write is blocked for a field, remove that field.\n"
        "- Prefer minimal edits.\n"
        f"operation={json.dumps(op, ensure_ascii=True)}\n"
        f"error={type(exc).__name__}: {str(exc)}\n"
        f"allowed_fields={json.dumps(describe_fields, ensure_ascii=True)}\n"
    )
    try:
        client = get_claude_client()
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=500,
            temperature=0,
            system=(
                "You fix malformed Salesforce mutation operations. "
                "Return strict JSON only, no markdown."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts: list[str] = []
        for part in response.content:
            if getattr(part, "type", "") == "text":
                text_parts.append(part.text)
        repaired = _extract_json_object("\n".join(text_parts).strip())
        if not isinstance(repaired, dict):
            return None, ""
        if str(repaired.get("op", "")).strip().lower() != str(op.get("op", "")).strip().lower():
            return None, ""
        if str(repaired.get("object", "")).strip() != object_name:
            return None, ""
        if fields and not repaired.get("fields") and str(op.get("op", "")).strip().lower() != "sobject_delete":
            return None, ""
        return repaired, "model-repaired malformed operation and retried"
    except Exception:
        return None, ""


def _build_observability_blob(events: list[dict[str, str]]) -> str:
    lines = ["Execution trace"]
    if not events:
        lines.append("- No execution steps were run.")
        return "```\n" + "\n".join(lines) + "\n```"
    for event in events:
        lines.append(
            f"- Step {event['step']} [{event['type']} | {event['status']}]: {event['reason']}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


def _is_salesforce_id(value: str) -> bool:
    return bool(SF_ID_RE.fullmatch(value.strip()))


def _is_salesforce_id_or_ref(value: str) -> bool:
    text = value.strip()
    return _is_salesforce_id(text) or bool(REF_TOKEN_RE.fullmatch(text))


def _operation_alias(op: dict[str, Any], idx: int) -> str:
    alias = str(op.get("id", "")).strip()
    if alias:
        return alias
    return f"op{idx + 1}"


def _parse_ref_alias(value: str) -> str | None:
    match = REF_TOKEN_RE.fullmatch(value.strip())
    if not match:
        return None
    return match.group(1)


def _validate_ref_token(
    record_id: str,
    aliases: dict[str, int],
    current_idx: int,
    op_number: int,
) -> None:
    alias = _parse_ref_alias(record_id)
    if alias is None:
        return
    if alias not in aliases:
        raise ValueError(f"Operation #{op_number} references unknown alias `{alias}`.")
    if aliases[alias] >= current_idx:
        raise ValueError(f"Operation #{op_number} must reference a prior operation, got `{alias}`.")


def _validate_field_refs(
    fields: Any,
    aliases: dict[str, int],
    current_idx: int,
    op_number: int,
) -> None:
    if not isinstance(fields, dict):
        return
    for key, value in fields.items():
        if not isinstance(value, str):
            continue
        alias = _parse_ref_alias(value)
        if alias is None:
            continue
        if alias not in aliases:
            raise ValueError(
                f"Operation #{op_number} field `{key}` references unknown alias `{alias}`."
            )
        if aliases[alias] >= current_idx:
            raise ValueError(
                f"Operation #{op_number} field `{key}` must reference a prior operation, got `{alias}`."
            )


def _resolve_operation_references(
    op: dict[str, Any],
    ref_values: dict[str, str],
    op_number: int,
) -> dict[str, Any]:
    resolved = dict(op)
    record_id = str(op.get("record_id", "")).strip()
    if record_id:
        resolved["record_id"] = _resolve_ref_value(
            value=record_id,
            ref_values=ref_values,
            context=f"Operation #{op_number} record_id",
        )
    fields = op.get("fields", {})
    if isinstance(fields, dict):
        resolved_fields: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, str):
                resolved_fields[key] = _resolve_ref_value(
                    value=value,
                    ref_values=ref_values,
                    context=f"Operation #{op_number} field `{key}`",
                )
            else:
                resolved_fields[key] = value
        resolved["fields"] = resolved_fields
    return resolved


def _resolve_ref_value(value: str, ref_values: dict[str, str], context: str) -> str:
    alias = _parse_ref_alias(value)
    if alias is None:
        return value
    if alias not in ref_values:
        raise ValueError(f"{context} references `{alias}` but no id is available yet.")
    return ref_values[alias]


def _register_operation_reference(
    alias: str,
    op: dict[str, Any],
    op_result: Any,
    ref_values: dict[str, str],
) -> None:
    op_type = str(op.get("op", "")).strip().lower()
    if op_type == "sobject_create":
        if isinstance(op_result, dict):
            created_id = str(op_result.get("id", "")).strip()
            if created_id:
                ref_values[alias] = created_id
        return
    if op_type in {"sobject_update", "sobject_delete"}:
        record_id = str(op.get("record_id", "")).strip()
        if _is_salesforce_id(record_id):
            ref_values[alias] = record_id


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        candidate = _extract_first_json_object_text(text)
        if candidate is None:
            raise RuntimeError("Invalid JSON from model repair output.") from exc
        return json.loads(candidate)


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


def _soql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _describe_field_names(sf: Any, object_name: str) -> set[str]:
    try:
        describe = sf.__getattr__(object_name).describe()
    except Exception:
        return set()
    raw_fields = describe.get("fields", []) if isinstance(describe, dict) else []
    names: set[str] = set()
    if isinstance(raw_fields, list):
        for item in raw_fields:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                names.add(name)
    return names


def _resolve_lookup_field_name(
    object_name: str,
    requested_field: str,
    available_fields: set[str],
) -> str:
    if not available_fields:
        return requested_field
    if requested_field in available_fields:
        return requested_field
    if requested_field.lower() == "name":
        for fallback in ("Name", "Subject", "CaseNumber"):
            if fallback in available_fields:
                return fallback
    if object_name.lower() == "case":
        for fallback in ("Subject", "CaseNumber"):
            if fallback in available_fields:
                return fallback
    return requested_field

