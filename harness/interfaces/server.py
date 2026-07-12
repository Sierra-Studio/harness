"""HTTP server exposing the agent loop as Server-Sent Events (SSE).

A thin, dependency-free wrapper (stdlib `http.server`) so browser/remote clients
can watch a turn unfold live — assistant text deltas plus every tool call and its
result — as the harness runs. The synchronous loop maps cleanly onto a
thread-per-request `ThreadingHTTPServer`.

Endpoints:
    POST /sessions  {"user_id": "..."}            -> {session_id, context_window, budget}
    POST /chat      {"session_id": "...",          -> text/event-stream of:
                     "message": "..."}                data: {"kind": "text"|"tool_start"|
                                                            "tool_result"|"final", ...}
                                                       ...
                                                       data: [DONE]

Run with:  uv run harness serve        (host/port via HARNESS_HTTP_HOST/PORT)
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..core import Harness
from ..llm import detect_provider, provider_label
from ..persistence import build_repository
from ..settings import Config, mcp_http_servers


def _event_to_dict(ev) -> dict:
    """Serialize a LoopEvent for the wire, expanding the final TurnResult."""
    if ev.kind == "final":
        return {"kind": "final", "result": dataclasses.asdict(ev.result)}
    if ev.kind == "text":
        return {"kind": "text", "text": ev.text}
    if ev.kind == "tool_start":
        return {"kind": "tool_start", "name": ev.name, "args": ev.args, "call_id": ev.call_id}
    if ev.kind == "tool_result":
        return {
            "kind": "tool_result",
            "name": ev.name,
            "content": ev.content,
            "call_id": ev.call_id,
        }
    return {"kind": ev.kind}


class _Handler(BaseHTTPRequestHandler):
    server_version = "Harness/1.0"

    @property
    def harness(self) -> Harness:
        return self.server.harness  # type: ignore[attr-defined]

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") == "/sessions":
            return self._create_session()
        if self.path.rstrip("/") == "/chat":
            return self._chat()
        self._send_json(404, {"error": f"no route {self.path}"})

    def _create_session(self) -> None:
        body = self._read_json()
        user_id = (body.get("user_id") or "demo-user").strip()
        session = self.harness.start_session(user_id)
        self._send_json(
            200,
            {
                "session_id": session.id,
                "context_window": session.context_window,
                "budget": session.token_budget or 0,
            },
        )

    def _chat(self) -> None:
        body = self._read_json()
        session_id = (body.get("session_id") or "").strip()
        message = body.get("message") or ""
        if not session_id or not message:
            return self._send_json(400, {"error": "session_id and message required"})
        try:
            session = self.harness.repo.get_session(session_id)
        except Exception as e:
            return self._send_json(404, {"error": f"unknown session: {e}"})

        # HTTP/1.0 (BaseHTTPRequestHandler default) closes the socket when the
        # handler returns, which is what gives the SSE client a clean EOF at the
        # end of the stream — so we deliberately do NOT advertise keep-alive.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()
        try:
            for ev in self.harness.run_turn_stream(session, message):
                self._sse(_event_to_dict(ev))
            self._sse_raw("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected mid-stream

    def _sse(self, payload: dict) -> None:
        self._sse_raw(json.dumps(payload, ensure_ascii=False))

    def _sse_raw(self, data: str) -> None:
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):  # quieter default logging
        return


def build_harness() -> Harness:
    """Construct a shared Harness and connect MCP HTTP servers, as the CLI does."""
    cfg = Config.from_env()
    h = Harness(cfg, provider=detect_provider(cfg), repo=build_repository(cfg), echo=False)
    for srv in mcp_http_servers():
        try:
            h.add_mcp_http(srv["url"], srv["name"], srv["headers"], oauth=srv.get("oauth"))
        except Exception as e:  # noqa: BLE001 — a bad MCP server shouldn't kill serve
            print(f"MCP '{srv['name']}' FAILED ({srv['url']}): {e}")
    return h


def serve(host: str = "127.0.0.1", port: int = 8800) -> int:
    h = build_harness()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.harness = h  # type: ignore[attr-defined]
    backend = "Postgres" if h.cfg.database_url else "in-memory"
    label = provider_label(h.cfg)
    print(
        f"Harness SSE server on http://{host}:{port} · repo={backend} · "
        f"provider={label} · model={h.cfg.provider.model}"
    )
    print("  POST /sessions {user_id}  ·  POST /chat {session_id, message} -> SSE")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        for client in h.tools.mcp_clients.values():
            with contextlib.suppress(Exception):
                client.stop()
    return 0
