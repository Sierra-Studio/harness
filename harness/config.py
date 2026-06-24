"""Configuration loaded from environment (.env optional)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency). Does not override real env vars."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split("#", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # provider
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    model: str = os.environ.get("HARNESS_MODEL", "anthropic/claude-sonnet-4-6")
    openrouter_base_url: str = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    # database — empty => in-memory repository
    database_url: str = os.environ.get("DATABASE_URL", "")
    # embeddings
    embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.environ.get(
        "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
    )
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = _int("EMBEDDING_DIM", 1536)
    # budgets / loop
    token_budget_per_session: int = _int("TOKEN_BUDGET_PER_SESSION", 500_000)
    response_reserve_tokens: int = _int("RESPONSE_RESERVE_TOKENS", 4_000)
    max_steps: int = _int("MAX_STEPS", 25)
    checkpoint_every_user_turns: int = _int("CHECKPOINT_EVERY_USER_TURNS", 20)
    skill_induction_every_sessions: int = _int("SKILL_INDUCTION_EVERY_SESSIONS", 10)
    summary_keep_ratio: float = _float("SUMMARY_KEEP_RATIO", 0.10)
    # fallback context window when the provider can't report one
    default_context_window: int = _int("DEFAULT_CONTEXT_WINDOW", 128_000)


def load_config() -> Config:
    return Config()


def mcp_http_servers() -> list[dict]:
    """Remote HTTP MCP servers declared in the environment.

    MCP_HTTP_SERVERS = "name=url, name2=url2"
    Per-server bearer token (optional): MCP_<NAME>_TOKEN
    """
    raw = os.environ.get("MCP_HTTP_SERVERS", "").strip()
    servers: list[dict] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, _, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        token = os.environ.get(f"MCP_{name.upper()}_TOKEN", "").strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        servers.append({"name": name, "url": url, "headers": headers})
    return servers
