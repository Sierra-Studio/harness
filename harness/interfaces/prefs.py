"""Local persisted preferences for the `harness chat` interfaces.

Stored at `~/.harness/preferences.json`, alongside `mcp_servers.json` and the
OAuth token cache. These are USER PREFERENCES for the CLI/TUI applications,
not library config — `Harness`/`Config` never read this file themselves;
`chat()` in `cli.py` loads it explicitly and applies it as *defaults that a
real env var still overrides* (see `apply_defaults` below).

Fields (all optional; absent/None = built-in default applies):
    theme                    Textual theme name (TUI only)
    persona                  inline persona/identity text override
    system_prompt            raw system-prompt override (bypasses persona layering)
    model                    default model id
    token_budget              default per-session token budget (0 = unlimited)
    response_reserve_tokens  tokens reserved for the model's reply
"""

from __future__ import annotations

import json
import os
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

STORE = Path.home() / ".harness" / "preferences.json"

DEFAULTS: dict[str, Any] = {
    "theme": "",
    "persona": "",
    "system_prompt": "",
    "model": "",
    "token_budget": None,
    "response_reserve_tokens": None,
    "permission_mode": "",  # "auto" | "manual" ("" = built-in / env default)
}


def load() -> dict[str, Any]:
    try:
        data = json.loads(STORE.read_text())
    except Exception:
        data = {}
    return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}


def save(**updates: Any) -> None:
    current = load()
    current.update({k: v for k, v in updates.items() if k in DEFAULTS})
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(current, indent=2))


def apply_defaults(cfg: Any, *, from_dotenv: AbstractSet[str] = frozenset()) -> Any:
    """Layer saved preferences onto `cfg` as defaults. Precedence: a real
    shell env var always wins over a saved preference (matching
    `Config.from_env`'s own "env is explicit, opt-in" rule); a saved
    preference wins over a `.env`-file default, since typing `/model ...` is a
    more specific, more recent, more intentional action than whatever a
    checked-in `.env` happens to default to.

    `from_dotenv`: the set of env var names that came from `_load_dotenv`
    (see `cli.py`) rather than the caller's real environment — pass
    `cli._DOTENV_KEYS`. Without it, every `.env` value looks like a real env
    var and permanently outranks any saved preference.

    Returns a new Config (dataclasses.replace); `cfg` itself is untouched.
    """
    import dataclasses

    def real_env(name: str) -> bool:
        return bool(os.environ.get(name)) and name not in from_dotenv

    p = load()
    provider, loop, perms = cfg.provider, cfg.loop, cfg.permissions
    if p["model"] and not real_env("HARNESS_MODEL"):
        provider = dataclasses.replace(provider, model=p["model"])
    if p["token_budget"] is not None and not real_env("TOKEN_BUDGET_PER_SESSION"):
        loop = dataclasses.replace(loop, token_budget_per_session=p["token_budget"])
    if p["response_reserve_tokens"] is not None and not real_env("RESPONSE_RESERVE_TOKENS"):
        loop = dataclasses.replace(loop, response_reserve_tokens=p["response_reserve_tokens"])
    if p["permission_mode"] and not real_env("HARNESS_PERMISSION_MODE"):
        perms = dataclasses.replace(perms, mode=p["permission_mode"])
    if provider is cfg.provider and loop is cfg.loop and perms is cfg.permissions:
        return cfg
    return dataclasses.replace(cfg, provider=provider, loop=loop, permissions=perms)
