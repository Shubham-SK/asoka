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


class KnowledgeKind(str, enum.Enum):
    fact = "fact"
    rule = "rule"
    trend = "trend"
    hypothesis = "hypothesis"
    question = "question"


class ConfidenceTier(str, enum.Enum):
    strict_violation = "strict_violation"
    similar_past_approval = "similar_past_approval"
    observed_trend = "observed_trend"
    coworker_context = "coworker_context"


class KnowledgeLifecycleStatus(str, enum.Enum):
    active = "active"
    superseded = "superseded"


class KnowledgeQuestionStatus(str, enum.Enum):
    open = "open"
    resolved = "resolved"
    dismissed = "dismissed"
