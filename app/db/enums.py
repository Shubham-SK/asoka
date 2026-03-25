from __future__ import annotations

import enum


class WorkspaceStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class UserRole(str, enum.Enum):
    coworker = "coworker"
    end_user = "end_user"


class PlanStatus(str, enum.Enum):
    draft = "draft"
    pending_approval = "pending_approval"
    approved = "approved"
    denied = "denied"
    executed = "executed"
    failed = "failed"


class AuthType(str, enum.Enum):
    integration_credentials = "integration_credentials"
    oauth_user = "oauth_user"
