from functools import lru_cache
import json
import shlex

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    port: int = 8000
    app_base_url: str = "http://localhost:8000"
    database_url: str = "sqlite:///./app.db"

    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_app_token: str = ""
    slack_coworker_user_id: str = ""
    plan_notify_coworker_on_create: bool = True

    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-latest"
    anthropic_api_key: str = ""

    salesforce_domain: str = "login"
    salesforce_username: str = ""
    salesforce_password: str = ""
    salesforce_security_token: str = ""
    salesforce_oauth_client_id: str = ""
    salesforce_oauth_client_secret: str = ""
    salesforce_oauth_redirect_uri: str = ""
    oauth_state_secret: str = ""
    token_encryption_key: str = ""
    read_backend: str = "salesforce_api"
    salesforce_mcp_command: str = ""
    salesforce_mcp_args: str = ""
    salesforce_mcp_env_json: str = "{}"
    salesforce_mcp_init_timeout_seconds: int = 20
    salesforce_mcp_tool_timeout_seconds: int = 90

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token and self.slack_signing_secret)

    @property
    def salesforce_enabled(self) -> bool:
        return bool(
            self.salesforce_username and self.salesforce_password and self.salesforce_security_token
        )

    @property
    def salesforce_oauth_enabled(self) -> bool:
        return bool(
            self.salesforce_oauth_client_id
            and self.salesforce_oauth_client_secret
            and self.salesforce_oauth_redirect_uri
        )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def salesforce_mcp_enabled(self) -> bool:
        return bool(self.salesforce_mcp_command.strip())

    @property
    def salesforce_mcp_args_list(self) -> list[str]:
        value = self.salesforce_mcp_args.strip()
        if not value:
            return []
        return shlex.split(value)

    @property
    def salesforce_mcp_env(self) -> dict[str, str]:
        raw = self.salesforce_mcp_env_json.strip() or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
