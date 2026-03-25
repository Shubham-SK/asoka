from __future__ import annotations

import logging

from app.agent.service import run_read_agent
from app.config import Settings
from app.orchestrator.classifier import classify_message

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def handle_message(self, user_id: str, text: str, conversation_window: str = "") -> str:
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

        return self._handle_read_request(text=text, conversation_window=conversation_window)

    def _handle_context_edit(self, is_coworker: bool, text: str) -> str:
        if not is_coworker:
            return "Only the designated human coworker can edit context."

        return (
            "Context update path is wired for Phase 4. I captured your message intent and "
            f"will persist + summarize updates once the context store is added.\n\nInput: {text}"
        )

    def _handle_read_request(self, text: str, conversation_window: str = "") -> str:
        if not self.settings.salesforce_enabled:
            return (
                "Read request received, but Salesforce credentials are not configured yet. "
                "Please add SALESFORCE_USERNAME, SALESFORCE_PASSWORD, and SALESFORCE_SECURITY_TOKEN."
            )

        if not self.settings.llm_enabled or self.settings.llm_provider != "anthropic":
            return (
                "Read request received, but Claude is not configured. "
                "Set LLM_PROVIDER=anthropic, LLM_MODEL, and ANTHROPIC_API_KEY."
            )

        try:
            return run_read_agent(self.settings, text, conversation_window=conversation_window)
        except Exception as exc:
            logger.exception("LLM read agent failed: %s", exc)
            return (
                "I could not complete this read request. Please verify Salesforce credentials, "
                "ANTHROPIC_API_KEY, and that SALESFORCE_DOMAIN is set to 'login' or 'test'."
            )
