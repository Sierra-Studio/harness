"""Tool layer: built-in tools (always in the prompt) + Index Tools (via MCP,
discovered on demand). Built-ins are O(1) in the prompt; the potentially huge
set of MCP tools lives in `tool_index` and is reached through SearchTools.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .models import Session
from .repository import Repository
from .sandbox import SandboxBackend


class ToolRegistry:
    BUILTINS = ("SearchTools", "GetTools", "CallTool", "SearchSkills", "GetSkill", "Bash", "RenderUI")

    def __init__(self, repo: Repository, sandbox: SandboxBackend,
                 mcp_clients: Optional[dict] = None, *, bash_timeout: int = 60,
                 bash_max_output: int = 10_000):
        self.repo = repo
        self.sandbox = sandbox
        self.mcp_clients = mcp_clients or {}   # name -> McpClient
        self.bash_timeout = bash_timeout
        self.bash_max_output = bash_max_output

    # ---- specs sent to the model (only the 6 built-ins) ----
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
            fn("CallTool", "Invoke an external tool you found via SearchTools/"
               "GetTools. This is the ONLY way to actually run an external tool — "
               "you cannot call it directly by name. Pass its exact 'name' and an "
               "'arguments' object matching its input schema; omit optional "
               "parameters you don't have.",
               {"name": {"type": "string",
                         "description": "Exact tool name from SearchTools/GetTools."},
                "arguments": {"type": "object",
                              "description": "Arguments object per the tool's "
                              "input schema. Use {} if none are needed."}},
               ["name"]),
            fn("SearchSkills", "List or keyword-search the current user's saved "
               "skills (reusable procedures). Returns name + one-line summary "
               "for each. Omit the query to list all. Then call GetSkill to read "
               "the full procedure.",
               {"query": {"type": "string"}}, []),
            fn("GetSkill", "Fetch the full body (steps) of one saved skill by "
               "exact name (from SearchSkills results) before following it.",
               {"name": {"type": "string"}}, ["name"]),
            fn("Bash",
               "Run a shell command in your per-session sandbox. This is your "
               "UNIVERSAL FALLBACK: use it whenever no specialized tool fits the "
               "task but the operating system can solve it — file and text "
               "operations, git, curl/HTTP, package managers, running code or "
               "scripts, data wrangling, system inspection, and more. Prefer a "
               "purpose-built tool (via SearchTools) when one clearly fits; "
               "otherwise reach for Bash instead of giving up. The working "
               "directory persists across calls; exported env vars do not (prefix "
               "inline: VAR=value cmd). Returns exit code, stdout and stderr.",
               {"command": {"type": "string",
                            "description": "The shell command to execute."},
                "timeout": {"type": "integer",
                            "description": "Optional per-command timeout in seconds."}},
               ["command"]),
            fn("RenderUI",
               "Render a rich, interactive UI in the chat instead of plain "
               "markdown. Use for dashboards, metric/stat cards, comparisons, "
               "tables, charts, or simple forms — anything clearer as structured "
               "UI than prose. Pass a single 'root' node; containers nest other "
               "nodes via their 'children'. Whitelisted node types (field `type`):\n"
               "  Layout: Stack{gap?,children}, Row{gap?,align?,children}, "
               "Grid{cols?,children}, Card{title?,children}, "
               "Tabs{tabs:[{label,children}]}, Divider\n"
               "  Display: Heading{level?1-3,text}, Text{text,muted?}, "
               "Markdown{text}, Badge{text,tone?}, Stat{label,value,delta?}, "
               "Callout{tone?,title?,text}, Code{lang?,code}, "
               "Image{src,alt?,caption?} (src must be https:// or data:image/), "
               "Progress{value,max?,label?,tone?}\n"
               "  Data: Table{columns:[str],rows:[[str|num]]}, "
               "Chart{kind:'bar'|'line'|'pie',title?,series:[{label,value}]}\n"
               "  Interactive: Button{label,action,value?,tone?}, "
               "Select{action,options:[{label,value}]}, Input{action,placeholder?,multiline?}, "
               "Form{action,submitLabel?,children}\n"
               "tone is one of neutral|success|warning|danger. Interactive nodes "
               "carry an 'action' id; when the user clicks/selects/submits, you "
               "receive their choice as the next user message prefixed "
               "'[ui-action]'. Put Input/Select inside a Form to collect several "
               "fields and submit them together (keyed by each field's 'action'). "
               "Only these types render — anything else is dropped.",
               {"root": {"type": "object",
                         "description": "The root UINode (usually a Stack or Card)."}},
               ["root"]),
        ]

    # ---- dispatch ----
    def dispatch(self, session: Session, call: dict) -> dict:
        name, args, call_id = self._parse(call)
        try:
            if name == "SearchTools":
                content = self._search_tools(args)
            elif name == "GetTools":
                content = self._get_tool(args)
            elif name == "SearchSkills":
                content = self._search_skills(session, args)
            elif name == "CallTool":
                content = self._call_tool(args)
            elif name == "GetSkill":
                content = self._get_skill(session, args)
            elif name == "Bash":
                content = self._bash(session, args)
            elif name == "RenderUI":
                content = self._render_ui(args)
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

    def _search_skills(self, session: Session, args: dict) -> str:
        query = args.get("query")
        if query:
            skills = self.repo.search_skills(session.user_id, query, 5)
        else:
            skills = self.repo.list_skills(session.user_id)
        if not skills:
            return "No skills for this user yet."
        # progressive disclosure: name + one-line summary; body via GetSkill.
        return json.dumps([{"name": s.name, "summary": s.summary} for s in skills],
                          ensure_ascii=False)

    def _get_skill(self, session: Session, args: dict) -> str:
        skill = self.repo.get_skill(session.user_id, args.get("name", ""))
        if not skill:
            return "Skill not found. Use SearchSkills to list available skills."
        return json.dumps({"name": skill.name, "summary": skill.summary,
                           "body": skill.body}, ensure_ascii=False)

    def _bash(self, session: Session, args: dict) -> str:
        command = args.get("command", "")
        if not command.strip():
            return "ERROR: empty command."
        try:
            timeout = int(args.get("timeout") or self.bash_timeout)
        except (TypeError, ValueError):
            timeout = self.bash_timeout
        res = self.sandbox.exec(session.id, command, timeout=timeout)
        parts = [f"<exit_code>{res.exit_code}</exit_code>"]
        if res.cwd:
            parts.append(f"<cwd>{res.cwd}</cwd>")
        parts.append(f"<stdout>\n{res.stdout}\n</stdout>")
        if res.stderr:
            parts.append(f"<stderr>\n{res.stderr}\n</stderr>")
        return "\n".join(parts)

    def _render_ui(self, args: dict) -> str:
        """Display-only tool: the UI tree lives in the call arguments, which the
        frontend reads and renders. We don't execute anything here — just sanity
        check the payload and acknowledge so the model knows the render landed.
        """
        root = args.get("root")
        if not isinstance(root, dict) or not isinstance(root.get("type"), str):
            return ("ERROR: RenderUI needs a 'root' object with a 'type' field "
                    "(e.g. {\"root\": {\"type\": \"Stack\", \"children\": [...]}}).")
        return f"UI rendered (root: {root['type']})."

    def _call_tool(self, args: dict) -> str:
        """Execute an external (MCP index) tool. The model passes the tool's
        exact name plus an arguments object; we route to the owning MCP client.
        This is what makes discovered tools actually runnable.
        """
        name = args.get("name", "")
        if not name:
            return "ERROR: CallTool needs 'name' (from SearchTools/GetTools)."
        inner = args.get("arguments", {})
        if isinstance(inner, str):  # some models stringify the object
            try:
                inner = json.loads(inner or "{}")
            except json.JSONDecodeError:
                return "ERROR: 'arguments' must be a JSON object."
        if not isinstance(inner, dict):
            inner = {}
        return self._index_tool(name, inner)

    def _index_tool(self, name: str, args: dict) -> str:
        spec = self.repo.get_tool(name)
        if not spec:
            return f"Unknown tool '{name}'. Use SearchTools first."
        client = self.mcp_clients.get(spec.mcp_server)
        if not client:
            return f"MCP server '{spec.mcp_server}' is not connected."
        result = client.call_tool(name, args)
        return json.dumps(result, ensure_ascii=False)
