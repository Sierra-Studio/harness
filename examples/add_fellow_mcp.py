"""Plug the Fellow MCP server (https://fellow.app/mcp) into the harness over the
native Streamable-HTTP transport, authenticating with the built-in OAuth flow.

    uv run python examples/add_fellow_mcp.py

On first run a browser opens for Fellow login (OAuth 2.1 + PKCE); the token is
cached under ~/.harness/mcp-auth so later runs are non-interactive. If you have
a static bearer token instead, drop the `oauth=True` and pass headers=
{"Authorization": f"Bearer {token}"}.
"""
from __future__ import annotations

from harness.app import Harness

FELLOW_URL = "https://fellow.app/mcp"


def main() -> None:
    h = Harness()  # reads .env (Postgres if DATABASE_URL is set, else in-memory)

    # one line: connect over HTTP with interactive OAuth, index tools, enable dispatch
    client = h.add_mcp_http(FELLOW_URL, name="fellow", oauth=True)

    print("Connected to Fellow. Indexed tools into tool_index:")
    for t in client.list_tools():
        print(f"  - {t['name']}: {t.get('description', '')[:70]}")

    hits = h.repo.search_tools("meeting action items", 5)
    print(f"\nSearchTools('meeting action items') -> {[x.name for x in hits]}")

    client.stop()


if __name__ == "__main__":
    main()
