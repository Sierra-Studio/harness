"""Local persistence for runtime-added HTTP MCP servers.

Servers connected with `/mcp http ...` during a chat are saved here so they
reconnect automatically on the next launch — otherwise they'd live only for
that process (the `MCP_HTTP_SERVERS` env var is the only other auto-connect
source). Stored under `~/.harness/` alongside the OAuth token cache.

Only name/url/exposure are persisted — never tokens. Servers that need auth
belong in `MCP_HTTP_SERVERS` + `MCP_<NAME>_TOKEN`/`_OAUTH`, which this layer
does not touch.
"""

from __future__ import annotations

import json
from pathlib import Path

STORE = Path.home() / ".harness" / "mcp_servers.json"


def load() -> list[dict]:
    try:
        data = json.loads(STORE.read_text())
        return [s for s in data if isinstance(s, dict) and s.get("url")]
    except Exception:
        return []


def save(name: str, url: str, expose: str = "index") -> None:
    servers = [s for s in load() if s.get("url") != url]
    servers.append({"name": name, "url": url, "expose": expose})
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(servers, indent=2))


def remove(name_or_url: str) -> bool:
    servers = load()
    kept = [s for s in servers if s.get("name") != name_or_url and s.get("url") != name_or_url]
    if len(kept) == len(servers):
        return False
    STORE.write_text(json.dumps(kept, indent=2))
    return True
