"""Application settings, read from environment variables and the .env file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds variables later stages use (Postgres, Neo4j...)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr prints as '**********', so the token cannot leak via logs or repr().
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
