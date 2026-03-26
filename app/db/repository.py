from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.enums import AuthType, PlanStatus
from app.db.models import (
    Conversation,
    ConversationMessage,
    ExecutionPlan,
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
