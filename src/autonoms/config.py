"""Application settings, read from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds variables later stages use (Postgres, Neo4j...)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr prints as '**********', so the token cannot leak via logs or repr().
    github_token: SecretStr | None = None
    github_api_url: str = "https://api.github.com"

    # Where task workspaces (one cloned repo per task) live, and where to clone from.
    workspaces_dir: Path = Path("workspaces")
    github_clone_url: str = "https://github.com"

    # The web interface (a separate process that talks to the API).
    api_url: str = "http://127.0.0.1:8000"
    ui_port: int = 8765  # 8080 is commonly taken (Jenkins, proxies)

    # The local LLM. "9b-8k" is your 8k-context variant: faster and fits the GPU better.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b-8k"
    ollama_think: bool | None = None  # None = the model's default

    # When the conversation passes this many characters, older steps are collapsed (context.py).
    context_compact_chars: int = 12000

    # The sandbox that runs checks. Pinned tag: "latest" would change under you.
    sandbox_image: str = "python:3.13.7-slim"
    sandbox_timeout_seconds: int = 20
    sandbox_memory: str = "256m"

    # Search indexes (one per task) and the embedding model that powers them.
    indexes_dir: Path = Path("indexes")
    embedding_model: str = "BAAI/bge-small-en-v1.5"

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
