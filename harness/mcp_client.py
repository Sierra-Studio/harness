"""Minimal MCP client over the stdio transport (newline-delimited JSON-RPC 2.0).

Implements just what the harness needs: initialize, tools/list, tools/call.
Tools discovered here are written into `tool_index` (NOT the system prompt) so
that an arbitrary number of MCP tools never crowds out the memory budget.
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any, Optional


class McpClient:
    def __init__(self, command: list[str], name: str = ""):
        self.command = command
        self.name = name or (command[0] if command else "mcp")
        self._proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "harness", "version": "1.0.0"},
        })
        self._notify("notifications/initialized", {})

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, timeout: float = 30) -> Any:
        assert self._proc and self._proc.stdout
        rid = self._next_id()
        with self._lock:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            # read until we see the matching response id
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP server closed the connection")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(f"MCP error: {msg['error']}")
                    return msg.get("result")

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list", {})
        return result.get("tools", []) if result else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


def ingest_server(repo, embedder, client: McpClient) -> int:
    """List a server's tools and upsert them into tool_index. Returns count."""
    n = 0
    for tool in client.list_tools():
        name = tool.get("name")
        if not name:
            continue
        description = tool.get("description", "")
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        emb = embedder.embed(f"{name}\n{description}")
        repo.upsert_tool(client.name, name, description, schema, emb)
        n += 1
    return n
