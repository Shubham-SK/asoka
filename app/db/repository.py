from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.enums import (
    AuthType,
    ConfidenceTier,
    KnowledgeKind,
    KnowledgeLifecycleStatus,
    KnowledgeQuestionStatus,
    PlanStatus,
)
from app.db.models import (
    Conversation,
    ConversationMessage,
    ExecutionPlan,
    KnowledgeItem,
    SlackSalesforceIdentity,
    User,
    UserContextEntry,
    Workspace,
)


def ensure_workspace_and_user(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
) -> tuple[Workspace, User]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        workspace = Workspace(slack_team_id=workspace_id, name=workspace_id)
        db.add(workspace)
        db.flush()

    user = db.scalar(
        select(User).where(
            User.workspace_id == workspace.id,
            User.slack_user_id == slack_user_id,
        )
    )
    if user is None:
        user = User(workspace_id=workspace.id, slack_user_id=slack_user_id)
        db.add(user)
        db.flush()
    return workspace, user


def get_active_oauth_identity(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
) -> SlackSalesforceIdentity | None:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return None
    return db.scalar(
        select(SlackSalesforceIdentity).where(
            SlackSalesforceIdentity.workspace_id == workspace.id,
            SlackSalesforceIdentity.slack_user_id == slack_user_id,
            SlackSalesforceIdentity.auth_type == AuthType.oauth_user,
            SlackSalesforceIdentity.is_active.is_(True),
        )
    )


def upsert_oauth_identity(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
    salesforce_org_key: str,
    salesforce_user_id: str | None,
    salesforce_username: str | None,
    instance_url: str,
    access_token_encrypted: str,
    refresh_token_encrypted: str | None,
    token_expires_at: datetime | None,
    scopes: str | None,
    metadata_json: dict[str, Any] | None = None,
) -> SlackSalesforceIdentity:
    workspace, user = ensure_workspace_and_user(db, workspace_id=workspace_id, slack_user_id=slack_user_id)
    identity = db.scalar(
        select(SlackSalesforceIdentity).where(
            SlackSalesforceIdentity.workspace_id == workspace.id,
            SlackSalesforceIdentity.slack_user_id == slack_user_id,
            SlackSalesforceIdentity.salesforce_org_key == salesforce_org_key,
        )
    )
    if identity is None:
        identity = SlackSalesforceIdentity(
            workspace_id=workspace.id,
            user_id=user.id,
            slack_user_id=slack_user_id,
            salesforce_org_key=salesforce_org_key,
            auth_type=AuthType.oauth_user,
        )
        db.add(identity)

    identity.user_id = user.id
    identity.auth_type = AuthType.oauth_user
    identity.salesforce_user_id = salesforce_user_id
    identity.salesforce_username = salesforce_username
    identity.instance_url = instance_url
    identity.access_token_encrypted = access_token_encrypted
    if refresh_token_encrypted:
        identity.refresh_token_encrypted = refresh_token_encrypted
    identity.token_expires_at = token_expires_at
    identity.scopes = scopes
    identity.metadata_json = metadata_json or {}
    identity.is_active = True
    db.flush()
    return identity


def set_user_context_entry(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
    context_key: str,
    value: dict[str, Any],
) -> UserContextEntry:
    workspace, user = ensure_workspace_and_user(db, workspace_id=workspace_id, slack_user_id=slack_user_id)
    entry = db.scalar(
        select(UserContextEntry).where(
            UserContextEntry.workspace_id == workspace.id,
            UserContextEntry.slack_user_id == slack_user_id,
            UserContextEntry.context_key == context_key,
        )
    )
    if entry is None:
        entry = UserContextEntry(
            workspace_id=workspace.id,
            user_id=user.id,
            slack_user_id=slack_user_id,
            context_key=context_key,
        )
        db.add(entry)
    entry.user_id = user.id
    entry.context_value_json = value
    db.flush()
    return entry


def get_or_create_conversation(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
    slack_channel_id: str,
) -> Conversation:
    workspace, user = ensure_workspace_and_user(db, workspace_id=workspace_id, slack_user_id=slack_user_id)
    convo = db.scalar(
        select(Conversation).where(
            Conversation.workspace_id == workspace.id,
            Conversation.slack_user_id == slack_user_id,
            Conversation.slack_channel_id == slack_channel_id,
        )
    )
    if convo is None:
        convo = Conversation(
            workspace_id=workspace.id,
            user_id=user.id,
            slack_user_id=slack_user_id,
            slack_channel_id=slack_channel_id,
        )
        db.add(convo)
        db.flush()
    return convo


def append_conversation_message(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
    slack_channel_id: str,
    role: str,
    text: str,
    slack_ts: str | None = None,
) -> ConversationMessage:
    convo = get_or_create_conversation(
        db=db,
        workspace_id=workspace_id,
        slack_user_id=slack_user_id,
        slack_channel_id=slack_channel_id,
    )
    message = ConversationMessage(
        conversation_id=convo.id,
        workspace_id=convo.workspace_id,
        slack_user_id=slack_user_id,
        role=role,
        text=text,
        slack_ts=slack_ts,
    )
    db.add(message)
    db.flush()
    return message


def load_conversation_window(
    db: Session,
    workspace_id: str,
    slack_user_id: str,
    slack_channel_id: str,
    limit: int = 25,
) -> str:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return ""
    convo = db.scalar(
        select(Conversation).where(
            Conversation.workspace_id == workspace.id,
            Conversation.slack_user_id == slack_user_id,
            Conversation.slack_channel_id == slack_channel_id,
        )
    )
    if convo is None:
        return ""

    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == convo.id)
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    messages = list(reversed(db.scalars(stmt).all()))
    lines: list[str] = []
    for msg in messages:
        role = "assistant" if msg.role == "assistant" else "user"
        lines.append(f"{role}: {msg.text}")
    return "\n".join(lines)


def create_execution_plan(
    db: Session,
    workspace_id: str,
    requester_slack_user_id: str,
    summary: str,
    operations: list[dict[str, Any]],
    assumptions: list[dict[str, Any]] | None = None,
    safety_checks: list[dict[str, Any]] | None = None,
    status: PlanStatus = PlanStatus.pending_approval,
) -> ExecutionPlan:
    workspace, _ = ensure_workspace_and_user(
        db,
        workspace_id=workspace_id,
        slack_user_id=requester_slack_user_id,
    )
    plan = ExecutionPlan(
        workspace_id=workspace.id,
        requester_slack_user_id=requester_slack_user_id,
        status=status,
        summary=summary,
        assumptions_json=assumptions or [],
        operations_json=operations,
        safety_checks_json=safety_checks or [],
        plan_fingerprint=_compute_plan_fingerprint(summary=summary, operations=operations),
    )
    db.add(plan)
    db.flush()
    return plan


def get_execution_plan_for_workspace(
    db: Session,
    workspace_id: str,
    plan_id: str,
) -> ExecutionPlan | None:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return None
    return db.scalar(
        select(ExecutionPlan).where(
            ExecutionPlan.id == plan_id,
            ExecutionPlan.workspace_id == workspace.id,
        )
    )


def set_execution_plan_status(
    db: Session,
    workspace_id: str,
    plan_id: str,
    status: PlanStatus,
    reason: str = "",
    actor_slack_user_id: str = "",
    allowed_from_statuses: list[PlanStatus] | None = None,
) -> ExecutionPlan | None:
    plan = get_execution_plan_for_workspace(db=db, workspace_id=workspace_id, plan_id=plan_id)
    if plan is None:
        return None
    previous_status = plan.status
    if allowed_from_statuses and previous_status not in allowed_from_statuses:
        raise ValueError(
            f"Invalid status transition: {previous_status.value} -> {status.value} for plan {plan_id}"
        )
    if not _is_allowed_plan_transition(previous_status, status):
        raise ValueError(
            f"Disallowed status transition: {previous_status.value} -> {status.value} for plan {plan_id}"
        )
    plan.status = status
    audit_entry = {
        "actor_slack_user_id": actor_slack_user_id,
        "previous_status": previous_status.value,
        "status": status.value,
        "reason": reason,
        "at": datetime.utcnow().isoformat(),
    }
    checks = list(plan.safety_checks_json or [])
    checks.append(audit_entry)
    plan.safety_checks_json = checks
    db.flush()
    return plan


def list_pending_plan_summaries(
    db: Session,
    workspace_id: str,
    requester_slack_user_id: str | None = None,
    limit: int = 25,
) -> list[dict[str, str]]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return []

    stmt = (
        select(ExecutionPlan).where(
            ExecutionPlan.workspace_id == workspace.id,
            ExecutionPlan.status == PlanStatus.pending_approval,
        )
    )
    if requester_slack_user_id:
        stmt = stmt.where(ExecutionPlan.requester_slack_user_id == requester_slack_user_id)
    stmt = stmt.order_by(ExecutionPlan.created_at.asc()).limit(max(1, min(limit, 100)))
    plans = db.scalars(stmt).all()
    out: list[dict[str, str]] = []
    for plan in plans:
        out.append(
            {
                "id": plan.id,
                "requester_slack_user_id": plan.requester_slack_user_id,
                "summary": plan.summary,
                "created_at": plan.created_at.isoformat(),
                "status": plan.status.value,
                "rejection_reason": _extract_latest_reason(plan, PlanStatus.denied),
            }
        )
    return out


def list_plan_summaries(
    db: Session,
    workspace_id: str,
    statuses: list[PlanStatus] | None = None,
    requester_slack_user_id: str | None = None,
    limit: int = 25,
) -> list[dict[str, str]]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return []

    stmt = select(ExecutionPlan).where(ExecutionPlan.workspace_id == workspace.id)
    if statuses:
        stmt = stmt.where(ExecutionPlan.status.in_(statuses))
    if requester_slack_user_id:
        stmt = stmt.where(ExecutionPlan.requester_slack_user_id == requester_slack_user_id)

    stmt = stmt.order_by(ExecutionPlan.created_at.desc()).limit(max(1, min(limit, 100)))
    plans = db.scalars(stmt).all()

    out: list[dict[str, str]] = []
    for plan in plans:
        out.append(
            {
                "id": plan.id,
                "requester_slack_user_id": plan.requester_slack_user_id,
                "summary": plan.summary,
                "created_at": plan.created_at.isoformat(),
                "status": plan.status.value,
                "rejection_reason": _extract_latest_reason(plan, PlanStatus.denied),
            }
        )
    return out


def _extract_latest_reason(plan: ExecutionPlan, status: PlanStatus) -> str:
    checks = plan.safety_checks_json
    if not isinstance(checks, list):
        return ""
    for item in reversed(checks):
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")).strip().lower() != status.value:
            continue
        reason = str(item.get("reason", "")).strip()
        if reason:
            return reason
    return ""


def create_knowledge_item(
    db: Session,
    workspace_id: str,
    kind: KnowledgeKind,
    confidence_tier: ConfidenceTier,
    title: str,
    content: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    salesforce_org_key: str = "default",
    confidence_score: float = 0.5,
    canonical_key: str | None = None,
    sf_object_api_name: str | None = None,
    sf_field_api_name: str | None = None,
    question_status: KnowledgeQuestionStatus | None = None,
) -> KnowledgeItem:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        workspace = Workspace(slack_team_id=workspace_id, name=workspace_id)
        db.add(workspace)
        db.flush()
    item = KnowledgeItem(
        workspace_id=workspace.id,
        salesforce_org_key=salesforce_org_key,
        kind=kind,
        confidence_tier=confidence_tier,
        confidence_rank=_confidence_rank(confidence_tier),
        confidence_score=max(0.0, min(confidence_score, 1.0)),
        title=title.strip()[:500] or "Untitled knowledge item",
        content_json=content or {},
        provenance_json=provenance or {},
        canonical_key=(canonical_key or "").strip() or None,
        sf_object_api_name=(sf_object_api_name or "").strip() or None,
        sf_field_api_name=(sf_field_api_name or "").strip() or None,
        lifecycle_status=KnowledgeLifecycleStatus.active,
        question_status=question_status,
    )
    db.add(item)
    db.flush()
    return item


def list_knowledge_for_retrieval(
    db: Session,
    workspace_id: str,
    kinds: list[KnowledgeKind] | None = None,
    min_confidence_rank: int | None = None,
    sf_object_api_name: str | None = None,
    limit: int = 25,
) -> list[KnowledgeItem]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return []
    stmt = select(KnowledgeItem).where(
        KnowledgeItem.workspace_id == workspace.id,
        KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus.active,
    )
    if kinds:
        stmt = stmt.where(KnowledgeItem.kind.in_(kinds))
    if min_confidence_rank is not None:
        normalized_rank = min(max(1, int(min_confidence_rank)), 4)
        stmt = stmt.where(KnowledgeItem.confidence_rank >= normalized_rank)
    if sf_object_api_name:
        stmt = stmt.where(KnowledgeItem.sf_object_api_name == sf_object_api_name)
    stmt = stmt.order_by(KnowledgeItem.confidence_rank.desc(), KnowledgeItem.updated_at.desc()).limit(
        max(1, min(limit, 100))
    )
    return list(db.scalars(stmt).all())


def increment_knowledge_usage_counts(
    db: Session,
    workspace_id: str,
    knowledge_item_ids: list[str],
) -> int:
    if not knowledge_item_ids:
        return 0
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return 0
    unique_ids = list({item_id for item_id in knowledge_item_ids if str(item_id).strip()})
    if not unique_ids:
        return 0
    stmt = select(KnowledgeItem).where(
        KnowledgeItem.workspace_id == workspace.id,
        KnowledgeItem.id.in_(unique_ids),
        KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus.active,
    )
    items = list(db.scalars(stmt).all())
    for item in items:
        item.usage_count = int(item.usage_count or 0) + 1
    db.flush()
    return len(items)


def list_open_policy_questions(
    db: Session,
    workspace_id: str,
    limit: int = 25,
) -> list[KnowledgeItem]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return []
    stmt = (
        select(KnowledgeItem)
        .where(
            KnowledgeItem.workspace_id == workspace.id,
            KnowledgeItem.kind == KnowledgeKind.question,
            KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus.active,
            KnowledgeItem.question_status == KnowledgeQuestionStatus.open,
        )
        .order_by(KnowledgeItem.updated_at.desc())
        .limit(max(1, min(limit, 100)))
    )
    return list(db.scalars(stmt).all())


def list_knowledge_items(
    db: Session,
    workspace_id: str,
    limit: int = 25,
    include_superseded: bool = False,
    kinds: list[KnowledgeKind] | None = None,
    query: str | None = None,
) -> list[KnowledgeItem]:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return []
    stmt = select(KnowledgeItem).where(KnowledgeItem.workspace_id == workspace.id)
    if not include_superseded:
        stmt = stmt.where(KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus.active)
    if kinds:
        stmt = stmt.where(KnowledgeItem.kind.in_(kinds))
    stmt = stmt.order_by(KnowledgeItem.updated_at.desc()).limit(max(1, min(limit, 200)))
    items = list(db.scalars(stmt).all())
    text_query = (query or "").strip().lower()
    if not text_query:
        return items
    filtered: list[KnowledgeItem] = []
    for item in items:
        statement = str((item.content_json or {}).get("statement", "")).lower()
        if text_query in item.title.lower() or text_query in statement:
            filtered.append(item)
    return filtered


def get_knowledge_item_by_id(
    db: Session,
    workspace_id: str,
    knowledge_id: str,
) -> KnowledgeItem | None:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        return None
    return db.scalar(
        select(KnowledgeItem).where(
            KnowledgeItem.workspace_id == workspace.id,
            KnowledgeItem.id == knowledge_id,
        )
    )


def update_knowledge_item(
    db: Session,
    workspace_id: str,
    knowledge_id: str,
    *,
    title: str | None = None,
    statement: str | None = None,
    kind: KnowledgeKind | None = None,
    confidence_tier: ConfidenceTier | None = None,
    confidence_score: float | None = None,
    sf_object_api_name: str | None = None,
    sf_field_api_name: str | None = None,
    question_status: KnowledgeQuestionStatus | None = None,
    lifecycle_status: KnowledgeLifecycleStatus | None = None,
) -> KnowledgeItem | None:
    item = get_knowledge_item_by_id(db=db, workspace_id=workspace_id, knowledge_id=knowledge_id)
    if item is None:
        return None
    if title is not None:
        item.title = title.strip()[:500] or item.title
    if statement is not None:
        content = dict(item.content_json or {})
        content["statement"] = statement.strip()
        item.content_json = content
    if kind is not None:
        item.kind = kind
    if confidence_tier is not None:
        item.confidence_tier = confidence_tier
        item.confidence_rank = _confidence_rank(confidence_tier)
    if confidence_score is not None:
        item.confidence_score = max(0.0, min(float(confidence_score), 1.0))
    if sf_object_api_name is not None:
        item.sf_object_api_name = sf_object_api_name.strip() or None
    if sf_field_api_name is not None:
        item.sf_field_api_name = sf_field_api_name.strip() or None
    if question_status is not None:
        item.question_status = question_status
    if lifecycle_status is not None:
        item.lifecycle_status = lifecycle_status
    db.flush()
    return item


def delete_knowledge_item(
    db: Session,
    workspace_id: str,
    knowledge_id: str,
) -> bool:
    item = get_knowledge_item_by_id(db=db, workspace_id=workspace_id, knowledge_id=knowledge_id)
    if item is None:
        return False
    item.lifecycle_status = KnowledgeLifecycleStatus.superseded
    db.flush()
    return True


def resolve_or_supersede_by_canonical_key(
    db: Session,
    workspace_id: str,
    salesforce_org_key: str,
    kind: KnowledgeKind,
    canonical_key: str,
    replacement_content: dict[str, Any],
    replacement_title: str,
    confidence_tier: ConfidenceTier,
    confidence_score: float,
    provenance: dict[str, Any] | None = None,
) -> KnowledgeItem:
    workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
    if workspace is None:
        workspace = Workspace(slack_team_id=workspace_id, name=workspace_id)
        db.add(workspace)
        db.flush()
    existing = db.scalar(
        select(KnowledgeItem).where(
            KnowledgeItem.workspace_id == workspace.id,
            KnowledgeItem.salesforce_org_key == salesforce_org_key,
            KnowledgeItem.kind == kind,
            KnowledgeItem.canonical_key == canonical_key,
            KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus.active,
        )
    )
    supersedes_id: str | None = None
    if existing is not None:
        existing.lifecycle_status = KnowledgeLifecycleStatus.superseded
        supersedes_id = existing.id
    item = create_knowledge_item(
        db=db,
        workspace_id=workspace_id,
        kind=kind,
        confidence_tier=confidence_tier,
        confidence_score=confidence_score,
        title=replacement_title,
        content=replacement_content,
        provenance=provenance or {},
        salesforce_org_key=salesforce_org_key,
        canonical_key=canonical_key,
    )
    item.supersedes_id = supersedes_id
    db.flush()
    return item


def _is_allowed_plan_transition(from_status: PlanStatus, to_status: PlanStatus) -> bool:
    if from_status == to_status:
        return True
    allowed_transitions: dict[PlanStatus, set[PlanStatus]] = {
        PlanStatus.draft: {PlanStatus.pending_approval},
        PlanStatus.pending_approval: {PlanStatus.approved, PlanStatus.denied},
        PlanStatus.approved: {PlanStatus.executed, PlanStatus.failed},
        PlanStatus.denied: {PlanStatus.pending_approval},
        PlanStatus.executed: set(),
        PlanStatus.failed: {PlanStatus.pending_approval},
    }
    return to_status in allowed_transitions.get(from_status, set())


def _compute_plan_fingerprint(summary: str, operations: list[dict[str, Any]]) -> str:
    payload = {"summary": summary.strip(), "operations": operations}
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _confidence_rank(tier: ConfidenceTier) -> int:
    mapping = {
        ConfidenceTier.strict_violation: 1,
        ConfidenceTier.similar_past_approval: 2,
        ConfidenceTier.observed_trend: 3,
        ConfidenceTier.coworker_context: 4,
    }
    return mapping.get(tier, 1)
