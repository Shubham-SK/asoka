from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.enums import AuthType, PlanStatus, UserRole, WorkspaceStatus


def _uuid() -> str:
    return str(uuid4())


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slack_team_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[WorkspaceStatus] = mapped_column(
        Enum(WorkspaceStatus), default=WorkspaceStatus.active, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    users: Mapped[list["User"]] = relationship(back_populates="workspace")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="workspace")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("workspace_id", "slack_user_id", name="uq_users_workspace_slack_user"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    slack_display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.end_user, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="users")
    sf_identities: Mapped[list["SlackSalesforceIdentity"]] = relationship(back_populates="user")
    context_entries: Mapped[list["UserContextEntry"]] = relationship(back_populates="user")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class SlackSalesforceIdentity(Base):
    """
    Future-proof identity mapping:
    - Supports current integration credentials mode.
    - Supports future per-user OAuth with refresh tokens.
    """

    __tablename__ = "slack_salesforce_identities"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "slack_user_id",
            "salesforce_org_key",
            name="uq_sf_identity_user_org",
        ),
        Index("ix_sf_identity_workspace_user", "workspace_id", "slack_user_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    salesforce_org_key: Mapped[str] = mapped_column(String(128), default="default", nullable=False)
    auth_type: Mapped[AuthType] = mapped_column(
        Enum(AuthType), default=AuthType.integration_credentials, nullable=False
    )

    # OAuth-ready fields (nullable until OAuth flow exists).
    salesforce_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    salesforce_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    instance_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    access_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    scopes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Bookkeeping
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)  # low|medium|high
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    user: Mapped["User | None"] = relationship(back_populates="sf_identities")


class UserContextEntry(Base):
    __tablename__ = "user_context_entries"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "slack_user_id",
            "context_key",
            name="uq_user_context_workspace_user_key",
        ),
        Index("ix_user_context_workspace_user", "workspace_id", "slack_user_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    context_key: Mapped[str] = mapped_column(String(128), nullable=False)
    context_value_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    user: Mapped["User | None"] = relationship(back_populates="context_entries")


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "slack_user_id",
            "slack_channel_id",
            name="uq_conversation_workspace_user_channel",
        ),
        Index("ix_conversation_workspace_user", "workspace_id", "slack_user_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    slack_channel_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="conversations")
    user: Mapped["User | None"] = relationship(back_populates="conversations")
    messages: Mapped[list["ConversationMessage"]] = relationship(back_populates="conversation")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("ix_message_conversation_created", "conversation_id", "created_at"),
        Index("ix_message_workspace_user", "workspace_id", "slack_user_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|assistant
    text: Mapped[str] = mapped_column(Text, nullable=False)
    slack_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class ExecutionPlan(Base):
    __tablename__ = "execution_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    requester_slack_user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[PlanStatus] = mapped_column(Enum(PlanStatus), default=PlanStatus.draft, nullable=False)
    summary: Mapped[str] = mapped_column(String(500))
    assumptions_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    operations_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    safety_checks_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    plan_version: Mapped[int] = mapped_column(default=1, nullable=False)
    plan_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
