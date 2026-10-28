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

    postgres_connection_string: SecretStr | None = None

    @property
    def database_url(self) -> str:
        if self.postgres_connection_string is None:
            raise RuntimeError("POSTGRES_CONNECTION_STRING is not set in .env")
        url = self.postgres_connection_string.get_secret_value()
        # A plain "postgresql://" URL makes SQLAlchemy look for the old psycopg2
        # driver. We installed psycopg 3, so name it explicitly.
        return url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
