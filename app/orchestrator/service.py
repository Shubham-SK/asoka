from __future__ import annotations

import logging
from typing import Callable

from app.agent.service import run_read_agent
from app.config import Settings
from app.orchestrator.classifier import classify_message
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
    ) -> str:
        is_coworker = user_id == self.settings.slack_coworker_user_id
        classification = classify_message(
            text,
            is_coworker=is_coworker,
            settings=self.settings,
            conversation_window=conversation_window,
        )
        logger.info("intent=%s reason=%s", classification.intent, classification.reason)

        if classification.intent == "context_edit":
            return self._handle_context_edit(is_coworker=is_coworker, text=text)

        if classification.intent == "write_request":
            return (
                "I detected a write request. Phase 2 approval workflow is next: "
                "I will draft a plan and request coworker approval before any Salesforce write."
            )

        return self._handle_read_request(
            user_id=user_id,
            text=text,
            workspace_id=workspace_id,
            conversation_window=conversation_window,
            progress_callback=progress_callback,
        )

    def _handle_context_edit(self, is_coworker: bool, text: str) -> str:
        if not is_coworker:
            return "Only the designated human coworker can edit context."

        return (
            "Context update path is wired for Phase 4. I captured your message intent and "
            f"will persist + summarize updates once the context store is added.\n\nInput: {text}"
        )

    def _handle_read_request(
        self,
        user_id: str,
        text: str,
        workspace_id: str = "",
        conversation_window: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> str:
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
            return run_read_agent(
                self.settings,
                text,
                slack_user_id=user_id,
                workspace_id=workspace_id or "default",
                conversation_window=conversation_window,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.exception("LLM read agent failed: %s", exc)
            return (
                "I could not complete this read request. Please verify Salesforce credentials, "
                "ANTHROPIC_API_KEY, and that SALESFORCE_DOMAIN is set to 'login' or 'test'."
            )
