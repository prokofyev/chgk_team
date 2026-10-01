from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHGK_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    mode: Literal["real", "demo"] = "demo"
    gigachat_credentials: SecretStr | None = Field(default=None, exclude=True, repr=False)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat-2"
    verify_ssl: bool = Field(
        default=True,
        validation_alias=AliasChoices("CHGK_VERIFY_SSL", "GIGACHAT__VERIFY_SSL"),
    )
    ca_bundle: str | None = Field(default=None, exclude=True)
    search_enabled: bool = True
    max_active_sessions: int = Field(default=4, ge=1, le=32)
    max_saved_sessions: int = Field(default=32, ge=1, le=256)
    session_ttl: float = Field(default=1800, ge=60)
    model_concurrency: int = Field(default=2, ge=1, le=16)
    turn_timeout: float = Field(default=6, gt=0, le=6)
    max_turns: int = Field(default=24, ge=1, le=24)
    context_tokens: int = Field(default=32000, ge=8000)
    output_tokens: int = Field(default=1000, ge=300, le=2000)
    demo_delay: float = Field(default=1.6, ge=0, le=5)

    @model_validator(mode="after")
    def validate_capacity(self):
        if self.max_saved_sessions < self.max_active_sessions:
            raise ValueError("Saved capacity must cover active sessions")
        return self

    @property
    def real_ready(self) -> bool:
        return bool(self.gigachat_credentials and self.gigachat_credentials.get_secret_value())

    def public(self) -> dict:
        return self.model_dump(mode="json", exclude={"gigachat_scope"})
