"""Shared configuration for the Day 22 LangSmith lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
EVIDENCE_DIR = ROOT_DIR / "evidence"


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str
    embedding_model: str
    langsmith_api_key: str | None
    langsmith_project: str
    langsmith_endpoint: str
    prompt_v1_name: str
    prompt_v2_name: str


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    project = _first_env(
        "LANGCHAIN_PROJECT",
        "LANGSMITH_PROJECT",
        default="day22-langsmith-prompt-versioning",
    )
    prompt_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", project or "day22").strip("-").lower()

    return Settings(
        openai_api_key=_first_env("OPENAI_API_KEY", "MODEL_API_KEY"),
        openai_base_url=_first_env("OPENAI_BASE_URL", "OPENAI_API_BASE", "MODEL_BASE_URL"),
        llm_model=_first_env("OPENAI_MODEL", "DEFAULT_LLM_MODEL", "MODEL_NAME", default="gpt-5.4-mini"),
        embedding_model=_first_env(
            "OPENAI_EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
            default="text-embedding-3-small",
        ),
        langsmith_api_key=_first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        langsmith_project=project or "day22-langsmith-prompt-versioning",
        langsmith_endpoint=_first_env(
            "LANGCHAIN_ENDPOINT",
            "LANGSMITH_ENDPOINT",
            default="https://api.smith.langchain.com",
        ),
        prompt_v1_name=_first_env("PROMPT_V1_NAME", default=f"{prompt_prefix}-rag-prompt-v1"),
        prompt_v2_name=_first_env("PROMPT_V2_NAME", default=f"{prompt_prefix}-rag-prompt-v2"),
    )


def configure_langsmith() -> Settings:
    """Set LangSmith env vars before LangChain objects are created."""
    settings = load_settings()
    os.environ["LANGCHAIN_TRACING_V2"] = _first_env("LANGCHAIN_TRACING_V2", default="true") or "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    return settings


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    EVIDENCE_DIR.mkdir(exist_ok=True)


def require_api_settings(settings: Settings) -> None:
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.langsmith_api_key:
        missing.append("LANGSMITH_API_KEY")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {joined}. "
            "Copy .env.example to .env and fill in your real values."
        )


def make_chat_model(settings: Settings, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def make_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


if __name__ == "__main__":
    cfg = configure_langsmith()
    ensure_dirs()
    print("Config loaded successfully")
    print(f"   LangSmith project : {cfg.langsmith_project}")
    print(f"   LangSmith endpoint: {cfg.langsmith_endpoint}")
    print(f"   OpenAI endpoint   : {cfg.openai_base_url or 'default OpenAI endpoint'}")
    print(f"   Default LLM model : {cfg.llm_model}")
    print(f"   Embedding model   : {cfg.embedding_model}")
    print(f"   API key present   : {'yes' if cfg.openai_api_key else 'no'}")
    print(f"   LangSmith key     : {'yes' if cfg.langsmith_api_key else 'no'}")

