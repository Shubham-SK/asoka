from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.enums import AuthType
from app.db.models import (
    Conversation,
    ConversationMessage,
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
