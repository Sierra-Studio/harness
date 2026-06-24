"""Plug the Fellow MCP server (https://fellow.app/mcp) into the harness over the
native Streamable-HTTP transport — no npx / mcp-remote bridge needed.

    uv run python examples/add_fellow_mcp.py

Auth: Fellow is a hosted server that requires authorization. Provide a bearer
token via the FELLOW_TOKEN env var (obtained from Fellow). Servers that require
an interactive OAuth login instead can still be reached with the `mcp-remote`
bridge + add_mcp_stdio, but the native path here covers token-based auth.
"""
from __future__ import annotations

import os

from harness.app import Harness

FELLOW_URL = "https://fellow.app/mcp"


def main() -> None:
    h = Harness()  # reads .env (Postgres if DATABASE_URL is set, else in-memory)

    headers = {}
    token = os.environ.get("FELLOW_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # one line: connect over HTTP, index tools into tool_index, enable dispatch
    client = h.add_mcp_http(FELLOW_URL, name="fellow", headers=headers)

    print(f"Connected to Fellow. Indexed tools into tool_index:")
    for t in client.list_tools():
        print(f"  - {t['name']}: {t.get('description', '')[:70]}")

    hits = h.repo.search_tools("meeting action items", 5)
    print(f"\nSearchTools('meeting action items') -> {[x.name for x in hits]}")

    client.stop()


if __name__ == "__main__":
    main()
