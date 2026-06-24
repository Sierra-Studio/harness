"""MCP clients for the harness.

Two transports, same duck-typed interface (`name`, `list_tools()`,
`call_tool()`): `McpClient` for local stdio servers and `HttpMcpClient` for
remote Streamable-HTTP servers. Both implement just what the harness needs:
initialize, tools/list, tools/call. Discovered tools are written into
`tool_index` (NOT the system prompt) so an arbitrary number of MCP tools never
crowds out the memory budget.
"""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Any, Optional

_CLIENT_INFO = {"name": "harness", "version": "1.0.0"}
_PROTOCOL_VERSION = "2025-06-18"  # MCP Streamable HTTP


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


class HttpMcpClient:
    """MCP Streamable HTTP transport.

    A single endpoint receives POSTed JSON-RPC; the server replies either with
    `application/json` (one response) or `text/event-stream` (SSE carrying the
    response). Session continuity uses the `Mcp-Session-Id` header. Pass
    `headers` for auth, e.g. {"Authorization": "Bearer <token>"}.
    """

    def __init__(self, url: str, name: str = "", headers: Optional[dict] = None,
                 timeout: float = 60):
        self.url = url
        self.name = name or url
        self._extra = headers or {}
        self._timeout = timeout
        self._client = None
        self._id = 0
        self._session_id: Optional[str] = None
        self._protocol = _PROTOCOL_VERSION

    def start(self) -> None:
        import httpx  # optional dependency

        self._client = httpx.Client(timeout=self._timeout)
        result = self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        })
        if result and result.get("protocolVersion"):
            self._protocol = result["protocolVersion"]
        self._notify("notifications/initialized", {})

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json",
             "Accept": "application/json, text/event-stream",
             "MCP-Protocol-Version": self._protocol}
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        h.update(self._extra)  # auth headers win
        return h

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _request(self, method: str, params: dict) -> Any:
        rid = self._next_id()
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        assert self._client is not None
        with self._client.stream("POST", self.url, json=payload,
                                 headers=self._headers()) as r:
            r.raise_for_status()
            sid = r.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
            if "text/event-stream" in r.headers.get("content-type", ""):
                msg = self._find_response_in_sse(r.iter_lines(), rid)
                if msg is None:
                    raise RuntimeError("no JSON-RPC response in SSE stream")
            else:
                msg = json.loads(r.read())
        if "error" in msg:
            raise RuntimeError(f"MCP error: {msg['error']}")
        return msg.get("result")

    def _notify(self, method: str, params: dict) -> None:
        # notifications get a 202 Accepted with no body
        assert self._client is not None
        self._client.post(self.url, json={"jsonrpc": "2.0", "method": method,
                                          "params": params}, headers=self._headers())

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list", {})
        return result.get("tools", []) if result else []

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def stop(self) -> None:
        if self._client:
            try:  # best-effort session termination per spec
                if self._session_id:
                    self._client.request("DELETE", self.url, headers=self._headers())
            except Exception:
                pass
            self._client.close()
            self._client = None

    @staticmethod
    def _find_response_in_sse(lines, rid):
        """Scan SSE `data:` lines for the JSON-RPC message matching `rid`."""
        for raw in lines:
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            try:
                msg = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                return msg
        return None


def ingest_server(repo, client) -> int:
    """List a server's tools and upsert them into tool_index. Returns count.

    Tools are stored in Postgres and searched by keyword (full-text) — no
    embeddings involved.
    """
    n = 0
    for tool in client.list_tools():
        name = tool.get("name")
        if not name:
            continue
        description = tool.get("description", "")
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        repo.upsert_tool(client.name, name, description, schema)
        n += 1
    return n
