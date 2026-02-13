"""Runtime settings for pipeline."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Environment-backed runtime settings."""

    model_config = ConfigDict(frozen=True)

    dry_run: bool = Field(default=True)
    environment: str = Field(default="dev")
    timezone: str = Field(default="America/Chicago")
    iem_base_url: str = Field(default="https://mesonet.agron.iastate.edu")
    kalshi_api_base_url: str = Field(default="https://api.elections.kalshi.com")



def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}



def load_settings(env_file: str = ".env") -> Settings:
    """Load settings from environment and optional .env file."""

    load_dotenv(dotenv_path=Path(env_file), override=False)
    from os import getenv

    return Settings(
        dry_run=_to_bool(getenv("DRY_RUN"), default=True),
        environment=getenv("ENVIRONMENT", "dev"),
        timezone=getenv("TIMEZONE", "America/Chicago"),
        iem_base_url=getenv("IEM_BASE_URL", "https://mesonet.agron.iastate.edu"),
        kalshi_api_base_url=getenv("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com"),
    )
