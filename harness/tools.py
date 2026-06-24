"""Tool layer: built-in tools (always in the prompt) + Index Tools (via MCP,
discovered on demand). Built-ins are O(1) in the prompt; the potentially huge
set of MCP tools lives in `tool_index` and is reached through SearchTools.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .embeddings import Embedder
from .models import Session
from .repository import Repository
from .sandbox import SandboxBackend


class ToolRegistry:
    BUILTINS = ("SearchTools", "GetTools", "GetSkills", "Bash")

    def __init__(self, repo: Repository, embedder: Embedder, sandbox: SandboxBackend,
                 mcp_clients: Optional[dict] = None):
        self.repo = repo
        self.embedder = embedder
        self.sandbox = sandbox
        self.mcp_clients = mcp_clients or {}   # name -> McpClient

    # ---- specs sent to the model (only the 4 built-ins) ----
    def builtin_specs(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props,
                               "required": required}}}
        return [
            fn("SearchTools", "Keyword search over available external tools. Use "
               "this to find a tool by describing what you need.",
               {"query": {"type": "string"},
                "k": {"type": "integer", "default": 8}}, ["query"]),
            fn("GetTools", "Fetch the full input schema of one tool by exact name "
               "(from SearchTools results) before calling it.",
               {"name": {"type": "string"}}, ["name"]),
            fn("GetSkills", "List or recall the current user's saved skills "
               "(reusable procedures). Optionally filter by a query.",
               {"query": {"type": "string"}}, []),
            fn("Bash", "Run a shell command inside the session sandbox.",
               {"command": {"type": "string"}}, ["command"]),
        ]

    # ---- dispatch ----
    def dispatch(self, session: Session, call: dict) -> dict:
        name, args, call_id = self._parse(call)
        try:
            if name == "SearchTools":
                content = self._search_tools(args)
            elif name == "GetTools":
                content = self._get_tool(args)
            elif name == "GetSkills":
                content = self._get_skills(session, args)
            elif name == "Bash":
                content = self._bash(session, args)
            else:
                content = self._index_tool(name, args)
        except Exception as e:  # tools must never crash the loop
            content = f"ERROR running {name}: {e}"
        return {"tool_call_id": call_id, "name": name, "content": content}

    @staticmethod
    def _parse(call: dict):
        # supports both OpenAI/OpenRouter format and a simplified one
        call_id = call.get("id") or call.get("tool_call_id") or ""
        fn = call.get("function", call)
        name = fn.get("name", "")
        raw = fn.get("arguments", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or "{}")
            except json.JSONDecodeError:
                raw = {}
        return name, raw, call_id

    # ---- built-in implementations ----
    def _search_tools(self, args: dict) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 8))
        hits = self.repo.search_tools(query, k)  # keyword search in Postgres
        if not hits:
            return "No tools found."
        return json.dumps([{"name": t.name, "description": t.description}
                           for t in hits], ensure_ascii=False)

    def _get_tool(self, args: dict) -> str:
        spec = self.repo.get_tool(args.get("name", ""))
        if not spec:
            return "Tool not found."
        return json.dumps({"name": spec.name, "description": spec.description,
                           "input_schema": spec.input_schema,
                           "server": spec.mcp_server}, ensure_ascii=False)

    def _get_skills(self, session: Session, args: dict) -> str:
        query = args.get("query")
        if query:
            skills = self.repo.search_skills(session.user_id,
                                             self.embedder.embed(query), 5)
        else:
            skills = self.repo.list_skills(session.user_id)
        if not skills:
            return "No skills for this user yet."
        # progressive disclosure: name + one-line summary (body loaded only if asked)
        return json.dumps([{"name": s.name, "summary": s.summary} for s in skills],
                          ensure_ascii=False)

    def _bash(self, session: Session, args: dict) -> str:
        res = self.sandbox.exec(session.id, args.get("command", ""))
        out = res.stdout
        if res.stderr:
            out += f"\n[stderr]\n{res.stderr}"
        return f"(exit {res.exit_code})\n{out}".strip()

    def _index_tool(self, name: str, args: dict) -> str:
        spec = self.repo.get_tool(name)
        if not spec:
            return f"Unknown tool '{name}'. Use SearchTools first."
        client = self.mcp_clients.get(spec.mcp_server)
        if not client:
            return f"MCP server '{spec.mcp_server}' is not connected."
        result = client.call_tool(name, args)
        return json.dumps(result, ensure_ascii=False)
