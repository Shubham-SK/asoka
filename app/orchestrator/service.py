from __future__ import annotations

import logging
from typing import Callable

from app.agent.service import run_read_agent
from app.agent.mcp_service import run_mcp_read_agent
from app.config import Settings
from app.evidence.ingestion import ingest_read_response_into_kb
from app.orchestrator.classifier import classify_message
from app.orchestrator.knowledge_agent import run_knowledge_agent
from app.orchestrator.plan_agent import run_plan_agent
from app.salesforce.oauth import build_oauth_start_url, has_user_oauth_identity

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def handle_message(
        self,
        user_id: str,
        text: str,
        workspace_id: str = "",
        conversation_window: str = "",
        progress_callback: Callable[[str], None] | None = None,
        plan_notification_callback: Callable[[str, str, str, str], None] | None = None,
        plan_status_notification_callback: Callable[[str, str, str, str, str, str], None] | None = None,
    ) -> str:
        is_coworker = user_id == self.settings.slack_coworker_user_id
        classification = classify_message(
            text,
            is_coworker=is_coworker,
            settings=self.settings,
            conversation_window=conversation_window,
        )
        logger.info("intent=%s reason=%s", classification.intent, classification.reason)

        if classification.intent == "knowledge_ingestion":
            return self._handle_manual_knowledge_ingestion(
                is_coworker=is_coworker,
                user_id=user_id,
                workspace_id=workspace_id or "default",
                text=text,
                conversation_window=conversation_window,
                parsed_intent=classification.intent,
                parsed_intent_reason=classification.reason,
                progress_callback=progress_callback,
            )
        if classification.intent == "knowledge_management":
            return self._handle_knowledge_management(
                is_coworker=is_coworker,
                user_id=user_id,
                workspace_id=workspace_id or "default",
                text=text,
                conversation_window=conversation_window,
                parsed_intent=classification.intent,
                parsed_intent_reason=classification.reason,
            )

        if classification.intent in {
            "write_request",
            "approval_response",
            "role_scope_query",
            "plan_management",
        }:
            return run_plan_agent(
                settings=self.settings,
                user_text=text,
                workspace_id=workspace_id or "default",
                requester_slack_user_id=user_id,
                is_coworker=is_coworker,
                parsed_intent=classification.intent,
                parsed_intent_reason=classification.reason,
                conversation_window=conversation_window,
                notify_pending_plan_callback=plan_notification_callback,
                notify_plan_status_callback=plan_status_notification_callback,
                progress_callback=progress_callback,
            )

        return self._handle_read_request(
            user_id=user_id,
            text=text,
            workspace_id=workspace_id,
            parsed_intent=classification.intent,
            parsed_intent_reason=classification.reason,
            conversation_window=conversation_window,
            progress_callback=progress_callback,
        )

    def _handle_read_request(
        self,
        user_id: str,
        text: str,
        workspace_id: str = "",
        parsed_intent: str = "",
        parsed_intent_reason: str = "",
        conversation_window: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        backend = self.settings.read_backend.strip().lower()
        if backend == "salesforce_mcp":
            if not self.settings.salesforce_mcp_enabled:
                return (
                    "Read backend is set to Salesforce MCP, but MCP is not configured. "
                    "Set SALESFORCE_MCP_COMMAND and SALESFORCE_MCP_ARGS."
                )
            if not self.settings.llm_enabled or self.settings.llm_provider != "anthropic":
                return (
                    "Read request received, but Claude is not configured. "
                    "Set LLM_PROVIDER=anthropic, LLM_MODEL, and ANTHROPIC_API_KEY."
                )
            try:
                response = run_mcp_read_agent(
                    settings=self.settings,
                    user_text=text,
                    parsed_intent=parsed_intent,
                    parsed_intent_reason=parsed_intent_reason,
                    conversation_window=conversation_window,
                    progress_callback=progress_callback,
                )
                return response
            except Exception as exc:
                logger.exception("MCP read agent failed: %s", exc)
                return (
                    "I could not complete this MCP read request. Verify MCP server configuration "
                    "and local Salesforce CLI authorization."
                )

        if not self.settings.salesforce_enabled and not has_user_oauth_identity(
            slack_user_id=user_id,
            workspace_id=workspace_id or "default",
        ):
            if self.settings.salesforce_oauth_enabled:
                connect_url = build_oauth_start_url(
                    slack_user_id=user_id,
                    workspace_id=workspace_id or "default",
                    settings=self.settings,
                )
                return (
                    "Read request received, but you have not connected Salesforce yet.\n"
                    f"Connect here: {connect_url}"
                )
            return (
                "Read request received, but Salesforce credentials are not configured yet. "
                "Either configure SALESFORCE_USERNAME/SALESFORCE_PASSWORD/SALESFORCE_SECURITY_TOKEN "
                "or set SALESFORCE_OAUTH_CLIENT_ID/SALESFORCE_OAUTH_CLIENT_SECRET/"
                "SALESFORCE_OAUTH_REDIRECT_URI for per-user OAuth."
            )

        if not self.settings.llm_enabled or self.settings.llm_provider != "anthropic":
            return (
                "Read request received, but Claude is not configured. "
                "Set LLM_PROVIDER=anthropic, LLM_MODEL, and ANTHROPIC_API_KEY."
            )

        try:
            response = run_read_agent(
                self.settings,
                text,
                slack_user_id=user_id,
                workspace_id=workspace_id or "default",
                parsed_intent=parsed_intent,
                parsed_intent_reason=parsed_intent_reason,
                conversation_window=conversation_window,
                progress_callback=progress_callback,
            )
            return response
        except Exception as exc:
            logger.exception("LLM read agent failed: %s", exc)
            return (
                "I could not complete this read request. Please verify Salesforce credentials, "
                "ANTHROPIC_API_KEY, and that SALESFORCE_DOMAIN is set to 'login' or 'test'."
            )

    def _handle_manual_knowledge_ingestion(
        self,
        is_coworker: bool,
        user_id: str,
        workspace_id: str,
        text: str,
        conversation_window: str,
        parsed_intent: str,
        parsed_intent_reason: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
        if not is_coworker:
            return "Only the designated human coworker can invoke knowledge ingestion."
        source = conversation_window.strip() or text.strip()
        if not source:
            return "Knowledge ingestion skipped: no source text available."
        result = ingest_read_response_into_kb(
            settings=self.settings,
            workspace_id=workspace_id,
            user_text=text,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
            slack_user_id=user_id,
            progress_callback=progress_callback,
        )
        return result.message

    def _handle_knowledge_management(
        self,
        is_coworker: bool,
        user_id: str,
        workspace_id: str,
        text: str,
        conversation_window: str,
        parsed_intent: str,
        parsed_intent_reason: str,
    ) -> str:
        if not is_coworker:
            return "Only the designated human coworker can manage knowledge instances."
        return run_knowledge_agent(
            settings=self.settings,
            user_text=text,
            workspace_id=workspace_id,
            requester_slack_user_id=user_id,
            parsed_intent=parsed_intent,
            parsed_intent_reason=parsed_intent_reason,
            conversation_window=conversation_window,
        )
