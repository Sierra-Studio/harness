"""Composable configuration, grouped by concern.

Every field here is a static literal default — nothing reads `os.environ` at
import time or as a field default. `Config()` with zero arguments is always
env-free and side-effect-free.

Reading the environment is an explicit, opt-in action the CALLING APPLICATION
takes via `Config.from_env()` (or a sub-config's own `from_env()`) — never
something `Harness` or `Config`'s own defaults do for you.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, "").strip() or default)
    except ValueError:
        return default


def _budget(env: Mapping[str, str], name: str, default: int) -> int:
    """Token budget: 0 (or none/unlimited/inf/-1) means no limit."""
    raw = env.get(name, "").strip().lower()
    if raw in ("0", "none", "unlimited", "inf", "-1"):
        return 0
    try:
        return int(raw or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class ProviderConfig:
    name: str = ""  # explicit registry key (e.g. "azure", "openrouter"); never sniffed
    model: str = "deepseek/deepseek-v4-pro"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Azure AI Foundry (OpenAI-compatible v1 endpoint).
    azure_endpoint: str = ""
    azure_api_key: str = ""  # empty => Entra ID / managed identity
    azure_api_version: str = "preview"
    azure_client_id: str = ""  # user-assigned managed identity
    default_context_window: int = 128_000  # fallback when the provider can't report one

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ProviderConfig:
        e = env if env is not None else os.environ
        return cls(
            name=e.get("HARNESS_PROVIDER_NAME", "").strip(),
            model=e.get("HARNESS_MODEL", cls.model).strip() or cls.model,
            openrouter_api_key=e.get("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=e.get("OPENROUTER_BASE_URL", cls.openrouter_base_url).strip()
            or cls.openrouter_base_url,
            azure_endpoint=e.get("AZURE_AI_ENDPOINT", "").strip(),
            azure_api_key=e.get("AZURE_AI_API_KEY", "").strip(),
            azure_api_version=e.get("AZURE_AI_API_VERSION", cls.azure_api_version).strip()
            or cls.azure_api_version,
            azure_client_id=e.get("AZURE_CLIENT_ID", "").strip(),
            default_context_window=_int(e, "DEFAULT_CONTEXT_WINDOW", cls.default_context_window),
        )


@dataclass(frozen=True)
class LoopConfig:
    max_steps: int = 25
    max_tool_calls_per_step: int = 8  # 0 = unlimited
    max_tool_calls_per_turn: int = 40  # 0 = unlimited
    token_budget_per_session: int = 500_000  # 0 = unlimited
    response_reserve_tokens: int = 4_000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LoopConfig:
        e = env if env is not None else os.environ
        return cls(
            max_steps=_int(e, "MAX_STEPS", cls.max_steps),
            max_tool_calls_per_step=_int(
                e, "MAX_TOOL_CALLS_PER_STEP", cls.max_tool_calls_per_step
            ),
            max_tool_calls_per_turn=_int(
                e, "MAX_TOOL_CALLS_PER_TURN", cls.max_tool_calls_per_turn
            ),
            token_budget_per_session=_budget(
                e, "TOKEN_BUDGET_PER_SESSION", cls.token_budget_per_session
            ),
            response_reserve_tokens=_int(
                e, "RESPONSE_RESERVE_TOKENS", cls.response_reserve_tokens
            ),
        )


@dataclass(frozen=True)
class MemoryConfig:
    checkpoint_every_user_turns: int = 20
    skill_induction_every_sessions: int = 10
    # max skills listed in the system prompt; above this, the model falls back
    # to SearchSkills for the long tail (the hybrid).
    skills_in_prompt_limit: int = 30
    summary_keep_ratio: float = 0.10
    # persona: path to a PERSONA.md (empty => look for ./PERSONA.md, else default)
    persona_path: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MemoryConfig:
        e = env if env is not None else os.environ
        return cls(
            checkpoint_every_user_turns=_int(
                e, "CHECKPOINT_EVERY_USER_TURNS", cls.checkpoint_every_user_turns
            ),
            skill_induction_every_sessions=_int(
                e, "SKILL_INDUCTION_EVERY_SESSIONS", cls.skill_induction_every_sessions
            ),
            skills_in_prompt_limit=_int(
                e, "SKILLS_IN_PROMPT_LIMIT", cls.skills_in_prompt_limit
            ),
            summary_keep_ratio=_float(e, "SUMMARY_KEEP_RATIO", cls.summary_keep_ratio),
            persona_path=e.get("HARNESS_PERSONA_PATH", e.get("HARNESS_SOUL_PATH", "")).strip(),
        )


@dataclass(frozen=True)
class BashConfig:
    timeout: int = 60  # per-command seconds
    max_output: int = 10_000  # chars before head/tail elision

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BashConfig:
        e = env if env is not None else os.environ
        return cls(
            timeout=_int(e, "BASH_TIMEOUT", cls.timeout),
            max_output=_int(e, "BASH_MAX_OUTPUT", cls.max_output),
        )


@dataclass(frozen=True)
class Config:
    database_url: str = ""  # empty => in-memory repository (via build_repository helper)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    bash: BashConfig = field(default_factory=BashConfig)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a Config by reading os.environ (or an injected mapping, for
        tests). The CALLER decides when/whether to use this — Harness never
        calls it, and Config()'s own defaults never read the environment."""
        e = env if env is not None else os.environ
        return cls(
            database_url=e.get("DATABASE_URL", "").strip(),
            provider=ProviderConfig.from_env(e),
            loop=LoopConfig.from_env(e),
            memory=MemoryConfig.from_env(e),
            bash=BashConfig.from_env(e),
        )


def mcp_http_servers(env: Mapping[str, str] | None = None) -> list[dict]:
    """Remote HTTP MCP servers declared in the environment. Opt-in helper —
    call it explicitly (e.g. from `cli.py`); nothing here reads env implicitly.

    MCP_HTTP_SERVERS = "name=url, name2=url2"
    Per-server bearer token (optional): MCP_<NAME>_TOKEN
    Per-server interactive OAuth (optional): MCP_<NAME>_OAUTH=1
    """
    e = env if env is not None else os.environ
    raw = e.get("MCP_HTTP_SERVERS", "").strip()
    servers: list[dict] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, _, url = entry.partition("=")
        name, url = name.strip(), url.strip()
        token = e.get(f"MCP_{name.upper()}_TOKEN", "").strip()
        oauth = e.get(f"MCP_{name.upper()}_OAUTH", "").strip() in ("1", "true", "yes")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        servers.append({"name": name, "url": url, "headers": headers, "oauth": oauth})
    return servers


__all__ = [
    "BashConfig",
    "Config",
    "LoopConfig",
    "MemoryConfig",
    "ProviderConfig",
    "mcp_http_servers",
]
