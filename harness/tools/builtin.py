"""Tool layer: a uniform `Tool` abstraction covering both the built-in tools
and developer-supplied ones. Tools are composed into a `ToolRegistry` from a
single list (see `default_tools`), and each carries its own model-facing spec
plus an optional `guidance` snippet the system prompt assembles on demand.

The potentially huge set of external MCP tools is NOT enumerated here — it lives
in the repository's `tool_index` and is reached through SearchTools/GetTools and
run through CallTool (or dispatched directly by name via `call_index_tool`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..memory import Skills
from ..models import Session
from ..persistence import Repository
from ..settings import Config
from .sandbox import SandboxBackend

if TYPE_CHECKING:
    from .capabilities import ToolProvider


# The interface-supplied mid-turn prompt callback, the free-form analogue of
# Permissions.asker: `(question, meta) -> answer`. `meta` may carry
# {"options": [...]} for a fixed choice set. Installed on ToolRegistry.prompter
# by the interface (blocking in CLI/TUI); None when non-interactive.
Prompter = Callable[[str, dict], str]


@dataclass
class ToolContext:
    """Dependencies handed to a tool at dispatch time.

    `mcp_clients` is the SAME live dict the registry owns (mutated in place by
    Harness.add_mcp_*), so a tool always sees currently-connected servers.
    `prompter` is the interface's mid-turn human-input callback (see AskUser);
    None when no human is available (non-interactive/server).
    """

    repo: Repository
    sandbox: SandboxBackend
    mcp_clients: dict  # name -> McpClient (live reference)
    config: Config
    skills: Skills
    prompter: Prompter | None = None


class Tool:
    """A single tool: a model-facing spec plus a handler.

    Subclass and set the `name` / `description` / `parameters` (JSON-Schema)
    class attributes — `spec()` assembles the OpenAI function spec from them, so
    subclasses carry no spec boilerplate. Optionally set `guidance` with a
    system-prompt snippet (may reference sibling tools); it is composed into the
    prompt only when this tool is active. Implement `run` to do the work.
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON-Schema object (subclasses override; never mutated)
    guidance: str = ""

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        raise NotImplementedError


class ToolSuspend(Exception):
    """Control-flow signal a tool (or an AskUser prompter) raises to SUSPEND the
    turn for out-of-band human input, instead of returning a result.

    Unlike every other exception a tool may raise — which `ToolRegistry.dispatch`
    deliberately swallows into an ``ERROR running ...`` result so a buggy tool
    never crashes the loop — this one is RE-RAISED by dispatch and propagates out
    of `run_turn_stream`. That lets an async backend abort the generator exactly
    as it does for the permission `asker`, capture the pending call (its `call_id`
    is on the `tool_start` event already streamed), and later resume with
    ``resume_turn_stream(approved_call={..., "result": answer})``. Synchronous
    interfaces (CLI/TUI) don't raise this — their prompter blocks and returns the
    answer directly. Carries the `question`/`options` for the backend's use."""

    def __init__(self, question: str = "", options: list | None = None):
        super().__init__(question)
        self.question = question
        self.options = options or []


# A custom tool handler receives (session, arguments) and returns the tool
# result as a string (same contract the built-ins effectively use).
ToolHandler = Callable[[Session, dict], str]


def make_tool(
    name: str,
    description: str,
    parameters: dict,
    handler: ToolHandler,
    *,
    guidance: str = "",
) -> Tool:
    """Build a Tool from parts without subclassing.

    `parameters` is a JSON-Schema object, e.g.
    {"type": "object", "properties": {...}, "required": [...]}. The `handler`
    runs in-process as `handler(session, arguments) -> str` (JSON-encode
    structured data yourself). Need the ToolContext (repo/sandbox/mcp)? Subclass
    `Tool` instead.
    """

    class _FunctionTool(Tool):
        def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
            return handler(session, args)

    tool = _FunctionTool()
    tool.name = name
    tool.description = description
    tool.parameters = parameters
    tool.guidance = guidance
    return tool


# --------------------------------------------------------------------------
# Shared MCP index-tool dispatch — used by CallTool AND the registry fallback.
# --------------------------------------------------------------------------
def call_index_tool(ctx: ToolContext, name: str, args: dict) -> str:
    """Execute an external (MCP index) tool by exact name, routing to the owning
    MCP client. This is what makes discovered tools actually runnable."""
    spec = ctx.repo.get_tool(name)
    if not spec:
        return f"Unknown tool '{name}'. Use SearchTools first."
    client = ctx.mcp_clients.get(spec.mcp_server)
    if not client:
        return f"MCP server '{spec.mcp_server}' is not connected."
    result = client.call_tool(name, args)
    return json.dumps(result, ensure_ascii=False)


class McpProxyTool(Tool):
    """Exposes a single MCP server tool as a DIRECT, first-class Tool.

    By default MCP tools are second-class: enumerated only in the repo index and
    reached via SearchTools/GetTools/CallTool. For a small, always-relevant server
    (a sandbox, a focused domain API) that discovery hop is pure friction — the
    model should just see the tools. `add_mcp_*(..., expose="direct")` wraps each
    of a server's tools in one of these so its schema is sent to the model
    directly. `run` calls the owning client itself, so the proxy carries no
    dependency on the repo index. The tool's summary is surfaced as `guidance`
    too, so the prompt advertises it without the app editing its system prompt.
    """

    def __init__(self, client, spec: dict):
        self._client = client
        self.name = spec.get("name", "")
        self.description = spec.get("description", "")
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}
        summary = (self.description.strip().splitlines() or [self.name])[0]
        self.guidance = f"- {self.name}: {summary}"

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        return json.dumps(self._client.call_tool(self.name, args), ensure_ascii=False)


# --------------------------------------------------------------------------
# Built-in tools
# --------------------------------------------------------------------------
class SearchTools(Tool):
    name = "SearchTools"
    description = (
        "Keyword search over available external tools. Use this to find a tool "
        "by describing what you need."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 8}},
        "required": ["query"],
    }
    guidance = (
        "- SearchTools(query): keyword-search the catalog of external tools (from MCP "
        "servers). If a specialized external tool likely exists, find it here first — a "
        "purpose-built tool beats a raw shell command. Query in ENGLISH with broad "
        "capability terms and synonyms (the catalog is indexed in English), and if the "
        "first query returns nothing, REFORMULATE with related terms before giving up — "
        "e.g. for 'gravações/recordings de reunião' try 'meeting', 'meeting transcript', "
        "'calls', 'recordings'. Data that clearly lives in an external service (meetings, "
        "calendar, email, chat, tickets, docs) almost always has a tool — do NOT fall back "
        "to Bash for it just because one keyword missed; Bash cannot see a SaaS account."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 8))
        hits = ctx.repo.search_tools(query, k)  # keyword search in Postgres
        if not hits:
            return "No tools found."
        return json.dumps(
            [{"name": t.name, "description": t.description} for t in hits], ensure_ascii=False
        )


class GetTools(Tool):
    name = "GetTools"
    description = (
        "Fetch the full input schema of one tool by exact name (from SearchTools "
        "results) before calling it."
    )
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    guidance = (
        "- GetTools(name): fetch the full input schema of one external tool before calling it."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        spec = ctx.repo.get_tool(args.get("name", ""))
        if not spec:
            return "Tool not found."
        return json.dumps(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "server": spec.mcp_server,
            },
            ensure_ascii=False,
        )


class CallTool(Tool):
    name = "CallTool"
    description = (
        "Invoke an external tool you found via SearchTools/GetTools. This is the "
        "ONLY way to actually run an external tool — you cannot call it directly "
        "by name. Pass its exact 'name' and an 'arguments' object matching its "
        "input schema; omit optional parameters you don't have."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact tool name from SearchTools/GetTools.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments object per the tool's input schema. "
                "Use {} if none are needed.",
            },
        },
        "required": ["name"],
    }
    guidance = (
        "- CallTool(name, arguments): actually RUN an external tool. External tools are NOT "
        "callable directly by name — CallTool is the only way to invoke one. The flow is "
        "always SearchTools -> GetTools -> CallTool.\n"
        "# Closing the loop — once you've found a tool, CALL it\n"
        "- After GetTools returns a schema, your next action is CallTool(name, arguments) for "
        "that tool. Do NOT re-run SearchTools/GetTools for a tool you already found, and never "
        "try to invoke a tool by emitting its name directly or by stuffing arguments into "
        "SearchTools — only CallTool runs it.\n"
        "- Parameters whose schema marks them optional (a default, or a type that allows null) "
        "can be OMITTED — pass arguments={} if you have nothing to supply. Don't stall hunting "
        "for values you don't have; call the tool with what you know and refine only if the "
        "result tells you to.\n"
        "- Never shell out to compute the date or time: today's date is already given to you at "
        "the end of this prompt. Use it directly for any date parameter.\n"
        "- If you catch yourself repeating an identical call, stop and change approach rather "
        "than looping."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
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
        return call_index_tool(ctx, name, inner)


class SearchSkills(Tool):
    name = "SearchSkills"
    description = (
        "List or keyword-search the current user's saved skills (reusable "
        "procedures). Returns name + one-line summary for each. Omit the query to "
        "list all. Then call GetSkill to read the full procedure."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": [],
    }
    guidance = (
        "- SearchSkills(query): keyword-search saved skills — only needed when the in-prompt "
        "list is truncated or you want to search a large set; returns name + summary for each."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        query = args.get("query")
        if query:
            skills = ctx.skills.search(session.user_id, query, 5)
        else:
            skills = ctx.skills.list(session.user_id)
        if not skills:
            return "No skills for this user yet."
        # progressive disclosure: name + one-line summary; body via GetSkill.
        return json.dumps(
            [{"name": s.name, "summary": s.summary} for s in skills], ensure_ascii=False
        )


class GetSkill(Tool):
    name = "GetSkill"
    description = (
        "Fetch the full body (steps) of one saved skill by exact name (from "
        "SearchSkills results) before following it."
    )
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    guidance = (
        '- GetSkill(name): load the full steps of one saved skill (from the "Your saved '
        'skills" list below, when present) before following it. If a saved skill covers the '
        "task, prefer it over improvising."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        skill = ctx.skills.get(session.user_id, args.get("name", ""))
        if not skill:
            return "Skill not found. Use SearchSkills to list available skills."
        return json.dumps(
            {"name": skill.name, "summary": skill.summary, "body": skill.body}, ensure_ascii=False
        )


class Bash(Tool):
    name = "Bash"
    description = (
        "Run a shell command in your per-session sandbox. This is your UNIVERSAL "
        "FALLBACK: use it whenever no specialized tool fits the task but the "
        "operating system can solve it — file and text operations, git, curl/HTTP, "
        "package managers, running code or scripts, data wrangling, system "
        "inspection, and more. Prefer a purpose-built tool (via SearchTools) when "
        "one clearly fits; otherwise reach for Bash instead of giving up. The "
        "working directory persists across calls; exported env vars do not (prefix "
        "inline: VAR=value cmd). Returns exit code, stdout and stderr."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute."},
            "timeout": {
                "type": "integer",
                "description": "Optional per-command timeout in seconds.",
            },
        },
        "required": ["command"],
    }
    guidance = (
        "- Bash(command): run a shell command in your per-session sandbox — your universal "
        "fallback. If the task CAN be done with ordinary operating-system commands (files, "
        "text processing, git, http via curl, package managers, running code, data wrangling, "
        "system inspection) and no specialized tool fits, use Bash. Do not refuse or describe "
        "what you would do — actually run the command.\n"
        "# Using Bash well\n"
        "- Your working directory persists across Bash calls within a session; `cd` into a "
        "project once and later commands stay there.\n"
        "- Exported environment variables do NOT persist between calls (each call is a fresh "
        "process). Prefix them inline: `VAR=value some_command`.\n"
        "- Chain related steps with && to keep them atomic; check the reported exit code and "
        "stderr, and fix-and-retry on failure instead of fabricating output.\n"
        "- Keep output focused (use head/tail/grep/wc) — very large output is truncated."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        command = args.get("command", "")
        if not command.strip():
            return "ERROR: empty command."
        try:
            timeout = int(args.get("timeout") or ctx.config.bash.timeout)
        except (TypeError, ValueError):
            timeout = ctx.config.bash.timeout
        res = ctx.sandbox.exec(session.id, command, timeout=timeout)
        parts = [f"<exit_code>{res.exit_code}</exit_code>"]
        if res.cwd:
            parts.append(f"<cwd>{res.cwd}</cwd>")
        parts.append(f"<stdout>\n{res.stdout}\n</stdout>")
        if res.stderr:
            parts.append(f"<stderr>\n{res.stderr}\n</stderr>")
        return "\n".join(parts)


class Write(Tool):
    name = "Write"
    description = (
        "Create a file or completely overwrite an existing one with the given "
        "content. Parent directories are created as needed. Paths are relative to "
        "your working directory (the same one Bash uses) unless absolute. This "
        "REPLACES the entire file — to change part of an existing file use Edit "
        "instead, which is safer. Returns the path written and the line count."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to cwd or absolute)."},
            "content": {"type": "string", "description": "The full file content to write."},
        },
        "required": ["path", "content"],
    }
    guidance = (
        "- Write(path, content): create a new file or overwrite one wholesale. It writes "
        "the ENTIRE file, so only use it for new files or a deliberate full rewrite; for a "
        "targeted change to an existing file use Edit instead (it can't clobber the rest)."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        path = str(args.get("path", "")).strip()
        content = args.get("content")
        if not path:
            return "ERROR: 'path' is required."
        if not isinstance(content, str):
            return "ERROR: 'content' must be a string (use \"\" for an empty file)."
        try:
            written = ctx.sandbox.write_file(session.id, path, content)
        except (OSError, NotImplementedError) as e:
            return f"ERROR: could not write {path!r}: {e}"
        return f"Wrote {len(content.splitlines())} line(s) to {written}"


class Edit(Tool):
    name = "Edit"
    description = (
        "Replace an exact string in an existing file, leaving the rest untouched. "
        "'old_string' must appear EXACTLY (including whitespace/indentation) and be "
        "UNIQUE in the file, or the edit is rejected — include enough surrounding "
        "context to make it unique. Set 'replace_all' to true to replace every "
        "occurrence instead. Read the file first (e.g. with Bash `cat`) so your "
        "'old_string' matches. Paths resolve like Write's. To create a file or "
        "rewrite it entirely, use Write."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit (relative to cwd or absolute)."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match.",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    guidance = (
        "- Edit(path, old_string, new_string, replace_all?): make a surgical, in-place change "
        "to an existing file. old_string must match EXACTLY and be unique (add surrounding "
        "context) unless replace_all is true. Prefer Edit over Write for changing existing "
        "files — it can't accidentally wipe unrelated content. Read the file first so the "
        "match is exact."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        path = str(args.get("path", "")).strip()
        old = args.get("old_string")
        new = args.get("new_string")
        if not path:
            return "ERROR: 'path' is required."
        if not isinstance(old, str) or not isinstance(new, str):
            return "ERROR: 'old_string' and 'new_string' must both be strings."
        if old == new:
            return "ERROR: 'old_string' and 'new_string' are identical — nothing to change."
        try:
            content = ctx.sandbox.read_file(session.id, path)
        except FileNotFoundError:
            return f"ERROR: file not found: {path} (use Write to create it)."
        except (OSError, NotImplementedError) as e:
            return f"ERROR: could not read {path!r}: {e}"
        count = content.count(old)
        if count == 0:
            return f"ERROR: 'old_string' not found in {path}. Read the file and match it exactly."
        replace_all = bool(args.get("replace_all"))
        if count > 1 and not replace_all:
            return (
                f"ERROR: 'old_string' appears {count} times in {path}. Add surrounding context "
                "to make it unique, or set replace_all=true."
            )
        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        try:
            written = ctx.sandbox.write_file(session.id, path, updated)
        except (OSError, NotImplementedError) as e:
            return f"ERROR: could not write {path!r}: {e}"
        edits = count if replace_all else 1
        return f"Edited {written} ({edits} replacement{'s' if edits != 1 else ''})"


class RenderUI(Tool):
    name = "RenderUI"
    description = (
        "Render a rich, interactive UI in the chat instead of plain markdown. Use "
        "for dashboards, metric/stat cards, comparisons, tables, charts, or simple "
        "forms — anything clearer as structured UI than prose. Pass a single 'root' "
        "node; containers nest other nodes via their 'children'. Whitelisted node "
        "types (field `type`):\n"
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
        "tone is one of neutral|success|warning|danger. Interactive nodes carry an "
        "'action' id; when the user clicks/selects/submits, you receive their choice "
        "as the next user message prefixed '[ui-action]'. Put Input/Select inside a "
        "Form to collect several fields and submit them together (keyed by each "
        "field's 'action'). Only these types render — anything else is dropped."
    )
    parameters = {
        "type": "object",
        "properties": {
            "root": {
                "type": "object",
                "description": "The root UINode (usually a Stack or Card).",
            }
        },
        "required": ["root"],
    }
    guidance = (
        "- RenderUI(root): render a rich interactive UI (cards, tables, charts, forms) instead "
        "of plain markdown when structured UI is clearer than prose."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        # Display-only tool: the UI tree lives in the call arguments, which the
        # frontend reads and renders. We don't execute anything — just sanity
        # check the payload and acknowledge so the model knows the render landed.
        root = args.get("root")
        if not isinstance(root, dict) or not isinstance(root.get("type"), str):
            return (
                "ERROR: RenderUI needs a 'root' object with a 'type' field "
                '(e.g. {"root": {"type": "Stack", "children": [...]}}).'
            )
        return f"UI rendered (root: {root['type']})."


class AskUser(Tool):
    """Pause the turn to ask the human a question, returning their answer as the
    tool result. The general, prompt-driven human-in-the-loop tool (as opposed to
    the fixed permission gate): the model calls it at whatever checkpoint the
    user's instructions/persona set up. The actual prompting is delegated to the
    interface-supplied `ToolContext.prompter` callback — blocking in the CLI/TUI,
    or (for async backends) something that raises to suspend the turn, resumed via
    `resume_turn_stream(approved_call={..., "result": answer})`. When no prompter
    is installed (non-interactive/server), it returns a sentinel and the model
    proceeds, so a turn never hangs."""

    name = "AskUser"
    description = (
        "Pause and ask the human for a decision, approval, or missing input before "
        "continuing. Use this at a checkpoint your instructions (or the user) asked "
        "you to confirm — e.g. an approval pipeline, or 'confirm before deploying' — "
        "or when you're genuinely blocked on a choice only the human can make. Pass "
        "the 'question' as clear prose. Optionally pass 'options' (a list of choices, "
        "e.g. [\"approve\", \"reject\"]) when there is a fixed set of answers; omit it "
        "for free-form input. The user's answer comes back as this tool's result — "
        "incorporate it and continue. If no human is available you'll get a note "
        "saying so; then proceed with your best judgment."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to put to the human, as clear prose.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional fixed set of answer choices (e.g. "
                '["approve", "reject"]). Omit for free-form input.',
            },
        },
        "required": ["question"],
    }
    guidance = (
        "- AskUser(question, options?): pause mid-task and ask the human for a "
        "decision, approval, or missing input, then continue with their answer as "
        "the tool result. Reach for it whenever your instructions or the user set up "
        "a confirmation/approval checkpoint (e.g. 'confirm before deploying', an "
        "approval pipeline between steps), or when you're truly blocked on a choice "
        "only the human can make. Pass 'options' for a fixed set of choices; omit it "
        "for free text. Don't overuse it — only stop when a human decision is "
        "genuinely required, not for things you can reasonably decide yourself."
    )

    _NO_HUMAN = (
        "No human is available to answer right now; proceed with your best judgment "
        "and note the assumption you made."
    )

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        question = str(args.get("question", "")).strip()
        if not question:
            return "ERROR: 'question' is required."
        if ctx.prompter is None:
            return self._NO_HUMAN
        options = args.get("options")
        meta = {"options": options} if isinstance(options, list) and options else {}
        answer = ctx.prompter(question, meta)
        return answer if (isinstance(answer, str) and answer.strip()) else "(no answer given)"


class ExitPlanMode(Tool):
    """Only meaningful (and only registered — see `sync_plan_mode_tool`) while
    the harness is in plan mode. Calling it is gated by `Permissions.check()`
    exactly like any other tool call: `run()` only ever executes once a human
    has approved the plan, so it carries no approval/mode-transition logic of
    its own — it just acknowledges."""

    name = "ExitPlanMode"
    description = (
        "Present your implementation plan to the user for approval before writing "
        "any code. Call this once you've finished researching in plan mode and have "
        "a concrete, actionable plan. Pass the full plan as markdown in 'plan' — be "
        "specific about files and changes, not just intent. If the user approves, "
        "you leave plan mode and may start implementing (each subsequent tool call "
        "will need a one-off confirmation). If they reject it, this call's result "
        "carries their feedback — revise the plan and call ExitPlanMode again."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The full implementation plan, written as markdown.",
            }
        },
        "required": ["plan"],
    }
    # Deliberately no `guidance`: guidance snippets are baked into the system
    # prompt once, at Harness construction / set_persona(), and would go stale
    # across a runtime mode toggle (nothing calls set_persona() on /plan or
    # shift+tab). The live, always-fresh explanation is this tool's own
    # `description` (sent fresh every turn via tool_specs()) plus the
    # per-turn plan-mode instruction block AgentLoop injects into the prompt.
    guidance = ""

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        return (
            "Plan approved by the user. You are now OUT of plan mode (manual mode) — "
            "proceed with implementing the plan. Each subsequent tool call will need "
            "a one-off approval from the user before it runs."
        )


def default_tools() -> list[Tool]:
    """Fresh instances of every built-in tool, in prompt order. This is the
    default `tools` when none are passed to the Harness/ToolRegistry."""
    return [
        SearchTools(),
        GetTools(),
        CallTool(),
        SearchSkills(),
        GetSkill(),
        Bash(),
        Write(),
        Edit(),
        RenderUI(),
        AskUser(),
    ]


class ToolRegistry:
    """Holds the active tool set (a name -> Tool map) and dispatches calls.

    `tools` is a single mixed list: include an item to have it, omit it to
    not. Each item is either
      * a `Tool` — an inert leaf, registered directly, or
      * a `ToolProvider` — a capability module (a sandbox, an MCP server, a
        bundle of domain tools) that is `register()`-ed against a
        `ProviderHost` wrapping this registry, contributing one or more tools.
    Ordering matters: items are installed in list order, which is also the
    prompt-guidance order. `tools` sentinel values:
      * None (default) -> all built-ins (`default_tools()`), no providers
      * True            -> same as None (all built-ins)
      * False (or [])  -> nothing at all
      * a list          -> exactly those items, in order
    Names not in the map are treated as external MCP index tools and dispatched
    by name (SearchTools/GetTools surface them; CallTool is the canonical path).
    """

    def __init__(
        self,
        repo: Repository,
        sandbox: SandboxBackend,
        mcp_clients: dict | None = None,
        *,
        config: Config,
        skills: Skills,
        tools: Iterable[Tool | ToolProvider] | bool | None = None,
        on_provider_error: Callable[[ToolProvider, Exception], None] | None = None,
    ):
        self.repo = repo
        self.sandbox = sandbox
        self.mcp_clients = mcp_clients or {}  # name -> McpClient (mutated by add_mcp_*)
        self.config = config
        self.skills = skills
        # Mid-turn human-input callback (see AskUser / ToolContext.prompter). The
        # interface installs it (like Permissions.asker); None means no human, so
        # AskUser returns a sentinel instead of hanging.
        self.prompter: Prompter | None = None
        self.tools: dict[str, Tool] = {}
        self.providers: list[ToolProvider] = []  # successfully registered, for close()
        # None/True -> all built-ins; False/[] -> none; else exactly the given list.
        # True is accepted so the bool half of the annotation is total — without
        # it, tools=True type-checks but crashes iterating over a bool.
        if tools is None or tools is True:
            resolved: Iterable = default_tools()
        elif not tools:
            resolved = []
        else:
            resolved = tools
        host = None  # built lazily; only needed if a ToolProvider is present
        for item in resolved:
            from .capabilities import ProviderHost, ToolProvider  # local: avoid import cycle

            if isinstance(item, ToolProvider):
                if host is None:
                    host = ProviderHost(self, self.repo)
                try:
                    item.register(host)
                except Exception as e:
                    if not item.optional:
                        raise
                    if on_provider_error is not None:
                        on_provider_error(item, e)
                else:
                    self.providers.append(item)
            else:
                self.register(item)

    def active_tools(self) -> list[Tool]:
        return list(self.tools.values())

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        """Add a tool to the active set after construction.

        The supported way to extend the live registry — e.g. exposing an MCP
        server's tools as direct tools (see `add_mcp_*(expose="direct")`). Because
        `tool_specs()` reads the map at call time, a tool registered here is sent
        to the model on the next turn. Raises on a name clash unless `replace`.
        """
        if not tool.name:
            raise ValueError("tool has no name")
        if tool.name in self.tools and not replace:
            raise ValueError(f"Duplicate tool name '{tool.name}' in registry.")
        self.tools[tool.name] = tool

    def deregister(self, name: str) -> bool:
        """Remove a tool from the active set. Returns True if it was present.
        Symmetric counterpart to register() — same "effective on the model's
        next turn" contract, since tool_specs() reads the map live."""
        return self.tools.pop(name, None) is not None

    # ---- specs sent to the model ----
    def tool_specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def _context(self) -> ToolContext:
        return ToolContext(
            self.repo, self.sandbox, self.mcp_clients, self.config, self.skills, self.prompter
        )

    # ---- dispatch ----
    def dispatch(self, session: Session, call: dict) -> dict:
        name, args, call_id = self._parse(call)
        ctx = self._context()
        try:
            tool = self.tools.get(name)
            if tool is not None:
                content = tool.run(ctx, session, args)
            else:
                # Not an active tool -> treat as an external MCP index tool.
                content = call_index_tool(ctx, name, args)
        except ToolSuspend:
            # NOT an error: a deliberate "pause the turn for human input" signal
            # (see AskUser). Let it propagate so the backend can suspend/resume,
            # rather than swallowing it into an ERROR result below.
            raise
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


def sync_plan_mode_tool(tools: ToolRegistry, mode: str) -> None:
    """Keep ExitPlanMode's registration in `tools` in sync with `mode`: present
    only while mode == "plan". Idempotent.

    CAUTION: do not call this from inside a permission `asker` callback while
    an ExitPlanMode call is itself mid-dispatch — deregistering here would
    remove the tool from the registry before ToolRegistry.dispatch() looks it
    up for the very call being approved, turning "plan approved" into an
    "Unknown tool 'ExitPlanMode'" error. AgentLoop._run_steps calls this at the
    top of every step (i.e. only after the previous step's dispatches have all
    finished), which is the one call site that's safe unconditionally; Harness
    also calls it eagerly whenever it's safe to do so (construction time,
    Harness.set_permission_mode from a slash command / shift+tab — never
    mid-dispatch).
    """
    has = "ExitPlanMode" in tools.tools
    if mode == "plan" and not has:
        tools.register(ExitPlanMode())
    elif mode != "plan" and has:
        tools.deregister("ExitPlanMode")
