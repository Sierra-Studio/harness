# HTTP server

`harness/interfaces/server.py` exposes the agent loop over HTTP as Server-Sent Events
(SSE) — a thin, dependency-free wrapper over the stdlib `http.server` so
browser/remote clients can watch a turn unfold live (assistant text deltas
plus every tool call and its result) as the harness runs. The synchronous loop
maps cleanly onto a thread-per-request `ThreadingHTTPServer`.

Run it:

```bash
uv run harness serve            # host/port via HARNESS_HTTP_HOST / HARNESS_HTTP_PORT
```

## Endpoints

**`POST /sessions`**

```json
{"user_id": "..."}
```

→

```json
{"session_id": "...", "context_window": 128000, "budget": 500000}
```

**`POST /chat`**

```json
{"session_id": "...", "message": "..."}
```

→ `text/event-stream` of:

```
data: {"kind": "text", "text": "..."}
data: {"kind": "tool_start", "name": "...", "args": {...}, "call_id": "..."}
data: {"kind": "tool_result", "name": "...", "content": "...", "call_id": "..."}
data: {"kind": "final", "result": {"text": "...", "status": "ok", "steps": 3, "tokens_spent": 1234}}
data: [DONE]
```
