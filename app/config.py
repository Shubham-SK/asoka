from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
