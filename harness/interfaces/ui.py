"""Shared rendering for the `harness chat` interfaces.

Every visible element is built by a `*_renderable()` function that returns a
Rich renderable using concrete colors (no console-theme names), so the exact
same widget renders identically whether it's printed by the line-based CLI
(`cli.py`) or mounted into the full-screen Textual app (`tui.py`). Thin printer
functions wrap the builders for the line-based path.

Design: minimal chrome, one accent color, structure from indentation and
dim/bold weight rather than borders — the register of `git`/`gh`/`docker`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.box import SIMPLE_HEAD
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

LOGO_ART_PATH = Path(__file__).parent / "assets" / "_logo_art.txt"

# concrete colors (not console-theme names) so renderables are portable to Textual
C_ACCENT = "cyan"
C_DIM = "grey62"
C_OK = "green3"
C_ERR = "red3"
C_WARN = "yellow3"
C_LIGHT = "grey85"  # logo's light chevron/diamond
C_GOLD = "#c9a063"  # logo's gold chevron


def logo_mark() -> Text:
    """The nautical mark as a stylized inline glyph — light chevron, light
    diamond, gold chevron — matching the logo's colors."""
    t = Text()
    t.append("❮", style=f"bold {C_LIGHT}")
    t.append("◆", style=f"bold {C_LIGHT}")
    t.append("❯", style=f"bold {C_GOLD}")
    return t

console = Console(highlight=False)

PROMPT = "›"

_TONE = {"neutral": C_ACCENT, "success": C_OK, "warning": C_WARN, "danger": C_ERR}

# Slash commands grouped by intent. Each group is (title, glyph, rows); rows are
# (command, description). help_renderable() lays the groups out as cards; a flat
# SLASH_COMMANDS view is derived below for any consumer that wants the whole list.
SLASH_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    ("Session", "◆", [
        ("/session", "id, model, tokens spent"),
        ("/sessions", "list your recent sessions"),
        ("/resume [n|id]", "resume a session (most recent if no arg)"),
        ("/new", "start a fresh session, same user"),
        ("/clear", "clear the screen (keeps the session)"),
        ("/exit, /quit", "end the session"),
    ]),
    ("Conversation", "↺", [
        ("/retry", "re-run your last message"),
        ("/copy", "copy the last answer to the clipboard"),
        ("/save [file]", "save the transcript as markdown"),
    ]),
    ("Configure", "⚙", [
        ("/model [name]", "show/change model (saved as default)"),
        ("/budget [n|unlimited]", "show/change token budget (saved as default)"),
        ("/theme [name]", "change the color theme — ^P also works"),
        ("/persona [text|clear]", "show/set/reset the persona (persisted)"),
        ("/system-prompt [text|clear]", "show/set/reset a raw system-prompt override"),
    ]),
    ("Tools & skills", "⛬", [
        ("/tools", "list active tools"),
        ("/skills", "list saved skills"),
        ("/skills add <name> <summary> [body...]", "author a new skill"),
        ("/mcp", "list connected MCP servers"),
        ("/mcp http <url> [name] [--direct]", "connect a remote MCP server"),
        ("/mcp stdio <name> <cmd...> [--direct]", "connect a local stdio MCP server"),
    ]),
]

# Flat view, kept for compatibility with any consumer that wants every command.
SLASH_COMMANDS = [("/help", "show this help")] + [
    row for _title, _glyph, rows in SLASH_GROUPS for row in rows
]

CLI_COMMANDS = [
    ("harness chat [user]", "interactive session (/help once inside)"),
    ("harness chat --inline", "Claude-Code-style inline UI (native scroll/select/copy)"),
    ("harness chat --plain", "minimal line-based REPL (no boxed input)"),
    ("harness init-db", "apply schema.sql to DATABASE_URL"),
    ("harness serve [host] [port]", "HTTP server, streams turns as SSE"),
    ("harness add-skill <user_id> <name> <summary> [body]", "author a skill"),
    ("harness list-skills <user_id>", "list a user's saved skills"),
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _elide(text: str, cap: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _oneline(text: str, cap: int) -> str:
    """Collapse whitespace — including *escaped* \\n/\\t that survive inside JSON
    string values — and hard-truncate to a single line."""
    flat = (text or "").replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    flat = " ".join(flat.split())
    return flat if len(flat) <= cap else flat[: cap - 1] + "…"


def tool_display_name(name: str, args: dict) -> str:
    """CallTool proxies another tool — show the tool it's actually invoking."""
    if name == "CallTool" and isinstance(args.get("name"), str):
        return args["name"]
    return name


def tool_args_str(name: str, args: dict, cap: int = 72) -> str:
    """Compact `key=value` rendering of tool arguments (not raw JSON)."""
    if name == "CallTool" and isinstance(args.get("arguments"), dict):
        args = args["arguments"]
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, str):
            parts.append(f'{k}="{_oneline(v, 28)}"')
        elif isinstance(v, (dict, list)):
            parts.append(f"{k}={_oneline(json.dumps(v, ensure_ascii=False), 28)}")
        else:
            parts.append(f"{k}={v}")
    return _oneline(", ".join(parts), cap)


_QUERY_WORDS = {"show", "help", "?"}


def parse_command(line: str) -> tuple[str, list[str], str]:
    """Split a slash-command line into (cmd, tokenized args, raw remainder).

    Tokenizes the remainder with shell-like quoting (shlex) — needed by
    commands like `/mcp stdio <name> <cmd...>` or `/skills add <name>
    "<summary>"` — but falls back to a plain whitespace split if the text has
    unbalanced quotes, so a stray apostrophe in ordinary prose ("you're",
    "don't", "it's") can NEVER crash the REPL with an uncaught ValueError.

    `raw` is the exact, untouched text after the command word — commands that
    want verbatim prose rather than shell-tokenized args (`/persona`,
    `/system-prompt`) should use `raw`, not `" ".join(args)`, which is lossy
    (collapses whitespace) and depends on tokenization succeeding at all.
    """
    cmd, _, rest = line.partition(" ")
    try:
        args = shlex.split(rest)
    except ValueError:
        args = rest.split()
    return cmd.lower(), args, rest.strip()


def is_query(args: list[str]) -> bool:
    """True if a slash command's args are a query, not a value to set.

    Guards the freeform setters (/persona, /system-prompt, /model, /budget):
    without this, `/persona show` or `/persona help` — a completely reasonable
    guess at the syntax — would silently SET the persona to the literal text
    "show"/"help" and report success, since anything not recognized as a
    subcommand falls through to "set this as the new value". `/persona`
    (no args) is still the primary, documented way to view the current value;
    this just keeps a likely typo from corrupting state instead of erroring
    or displaying it.
    """
    return len(args) == 1 and args[0].lower() in _QUERY_WORDS


def parse_budget(raw: str) -> int | None:
    """Parse a `/budget` argument: "0"/"none"/"unlimited"/"inf"/"-1" -> 0
    (unlimited); a plain integer -> itself; anything else -> None (invalid),
    same vocabulary as the TOKEN_BUDGET_PER_SESSION env var."""
    raw = (raw or "").strip().lower()
    if raw in ("0", "none", "unlimited", "inf", "-1"):
        return 0
    try:
        return int(raw)
    except ValueError:
        return None


def is_error_result(content: str) -> bool:
    if (content or "").lstrip().startswith("ERROR"):
        return True
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(obj, dict) and obj.get("isError") is True


def summarize_result(content: str, cap: int = 110) -> str:
    """One clean line from a raw tool result: extract text from MCP-style
    payloads, count list results, strip JSON/escape noise."""
    s = (content or "").strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("content"), list):
        texts = [c.get("text", "") for c in obj["content"] if isinstance(c, dict) and c.get("type") == "text"]
        s = " ".join(t for t in texts if t) or s
    elif isinstance(obj, list):
        names = [str(o.get("name", "")) for o in obj if isinstance(o, dict) and o.get("name")]
        head = ": " + ", ".join(names[:4]) if names else ""
        s = f"{len(obj)} result{'' if len(obj) == 1 else 's'}{head}"
    return _oneline(s, cap)


def _command_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style=C_ACCENT, no_wrap=True)
    table.add_column(style=C_DIM, overflow="ellipsis", no_wrap=True)
    for cmd, desc in rows:
        # escape() so bracketed argument hints aren't swallowed as Rich markup.
        table.add_row(escape(cmd), escape(desc))
    return table


# ---------------------------------------------------------------------------
# builders — return renderables (shared by CLI + TUI)
# ---------------------------------------------------------------------------
def logo_renderable(path: str | Path | None = None) -> RenderableType | None:
    """Welcome-card logo from the bundled block-character art
    (`assets/_logo_art.txt`). Point `HARNESS_LOGO_PATH` at another art file, or
    set it to an empty string to drop the logo. Returns None when the file is
    absent."""
    if path is not None:
        target: str | Path = path
    else:
        target = os.environ.get("HARNESS_LOGO_PATH", LOGO_ART_PATH)
    if not target:
        return None
    p = Path(target)
    if not p.exists():
        return None
    art = p.read_text(encoding="utf-8").rstrip("\n")
    if not art:
        return None
    return Text(art, style=C_GOLD)


def welcome_renderable(
    *, version: str, user: str, provider_label: str, model: str, backend: str, session_id: str
) -> RenderableType:
    """A bordered welcome card with the current chat's metadata — shown once at
    startup by both the TUI and the line-based CLI."""
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=C_DIM, justify="left")
    grid.add_column()
    grid.add_row("user", user)
    grid.add_row("provider", provider_label)
    grid.add_row("model", model)
    grid.add_row("storage", backend)
    grid.add_row("session", str(session_id))
    hint = Text("/help for commands · /exit to quit", style=C_DIM)
    keys = Text("^P palette · ^L clear · ^C stop · ↑↓ history · Tab complete", style=C_DIM)
    title = logo_mark()
    title.append("  harness ", style=f"bold {C_ACCENT}")
    title.append(version, style=C_DIM)
    parts: list[Any] = []
    logo = logo_renderable()
    if logo is not None:
        parts += [Align.center(logo), Text("")]
    parts += [grid, Text(""), hint, keys]
    return Panel(
        Group(*parts),
        title=title,
        title_align="left",
        border_style=C_ACCENT,
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
    )


def mcp_connected_renderable(name: str, url: str, n_tools: int) -> RenderableType:
    t = Text()
    t.append("✓ ", style=C_OK)
    t.append(f"{name:<12}", style="bold")
    t.append(f"  {n_tools} tools · {url}", style=C_DIM)
    return t


def mcp_failed_renderable(name: str, url: str, error: Exception) -> RenderableType:
    t = Text()
    t.append("✗ ", style=C_ERR)
    t.append(f"{name:<12}", style="bold")
    t.append(f"  {url}", style=C_DIM)
    t.append(f"\n    {error}", style=C_ERR)
    return t


def session_renderable(session: Any) -> RenderableType:
    budget = f"{session.token_budget:,}" if session.token_budget else "unlimited"
    rows = [
        ("id", str(session.id)),
        ("model", str(session.model)),
        ("context window", f"{session.context_window:,}"),
        ("budget", budget),
        ("tokens spent", f"{session.tokens_spent:,}"),
    ]
    t = Text()
    for i, (label, value) in enumerate(rows):
        if i:
            t.append("\n")
        t.append(f"{label:<15} ", style=C_DIM)
        t.append(value)
    return t


def _help_group_card(title: str, glyph: str, rows: list[tuple[str, str]]) -> RenderableType:
    """One category of slash commands as a titled, bordered card."""
    grid = Table.grid(padding=(0, 2, 0, 0))
    grid.add_column(style=C_ACCENT, no_wrap=True)
    grid.add_column(style=C_DIM, overflow="fold")
    for cmd, desc in rows:
        # escape() so bracketed hints — [name], [n|id], [file] — render literally
        # instead of being swallowed as Rich markup tags.
        grid.add_row(escape(cmd), escape(desc))
    heading = Text()
    heading.append(f"{glyph} ", style=C_GOLD)
    heading.append(title, style=f"bold {C_LIGHT}")
    return Panel(
        grid,
        title=heading,
        title_align="left",
        border_style=C_DIM,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=True,
    )


def help_renderable() -> RenderableType:
    """Slash commands grouped into titled cards, stacked so each command and its
    description gets full width — reads at a glance instead of as one flat list."""
    header = logo_mark()
    header.append("  commands", style=f"bold {C_ACCENT}")
    cards = [_help_group_card(title, glyph, rows) for title, glyph, rows in SLASH_GROUPS]
    footer = Text(
        "Anything else is sent to the model.  Type / for autocomplete · "
        "end a line with \\ to continue it · /help anytime.",
        style=C_DIM,
    )
    return Group(header, Text(""), *cards, Text(""), footer)


def _ago(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def sessions_renderable(sessions: list, active_id: str = "") -> RenderableType:
    if not sessions:
        return Text("no sessions yet", style=C_DIM)
    table = _base_table("sessions")
    table.add_column("#", style=C_DIM, justify="right")
    table.add_column("id", style=C_ACCENT, no_wrap=True)
    table.add_column("subject", overflow="ellipsis", no_wrap=True, max_width=44)
    table.add_column("when", style=C_DIM, no_wrap=True)
    table.add_column("turns", style=C_DIM, justify="right")
    table.add_column("tokens", style=C_DIM, justify="right")
    for i, s in enumerate(sessions, 1):
        when = "current" if s.id == active_id else _ago(s.started_at)
        table.add_row(str(i), s.id[:8], s.subject or "—", when, str(s.turns), f"{s.tokens_spent:,}")
    return table


def transcript_markdown(turns: list) -> str:
    """Render a session's user/assistant turns as a markdown transcript."""
    out = ["# Transcript", ""]
    for t in turns:
        if t.role not in ("user", "assistant"):
            continue
        content = t.content if isinstance(t.content, str) else json.dumps(t.content, ensure_ascii=False)
        out.append(f"## {t.role}")
        out.append("")
        out.append(content)
        out.append("")
    return "\n".join(out)


def resolve_resume(repo: Any, user_id: str, arg: str = "") -> Any:
    """Resolve a `/resume` argument to a SessionSummary (or None). No arg → most
    recent; a number → that row from the newest-first list; else an id prefix."""
    recent = repo.list_sessions(user_id, limit=50)
    if not recent:
        return None
    arg = (arg or "").strip()
    if not arg:
        return recent[0]
    if arg.isdigit():
        i = int(arg) - 1
        return recent[i] if 0 <= i < len(recent) else None
    return next((s for s in recent if s.id.startswith(arg)), None)


def cli_help_renderable() -> RenderableType:
    return Group(
        Text("harness", style="bold"),
        Text(""),
        _command_table(CLI_COMMANDS),
        Text("\nuv run <command>, or: python -m harness.interfaces.cli <command>", style=C_DIM),
    )


def skills_renderable(skills: list) -> RenderableType:
    if not skills:
        return Text("no skills yet", style=C_DIM)
    table = _base_table("skills")
    table.add_column("name", style=C_ACCENT, no_wrap=True)
    table.add_column("origin", style=C_DIM, no_wrap=True)
    table.add_column("summary", overflow="ellipsis", no_wrap=True, max_width=60)
    for s in skills:
        table.add_row(s.name, s.origin, _elide(s.summary, 60))
    return table


def _tool_source(t: Any) -> str:
    """Which MCP server a direct-exposed tool (McpProxyTool) came from, else
    'built-in'. Duck-typed on `_client.name` so this needs no import of the
    tool classes (avoids a tools -> interfaces coupling)."""
    client = getattr(t, "_client", None)
    name = getattr(client, "name", None) if client is not None else None
    return name or "built-in"


def tools_renderable(tools: list) -> RenderableType:
    """Grouped by source: built-ins first, then each MCP server (for
    direct-exposed tools) alphabetically — so it's clear which server a tool
    is actually linked to."""
    groups: dict[str, list] = {}
    for t in tools:
        groups.setdefault(_tool_source(t), []).append(t)

    parts: list[Any] = []
    for i, source in enumerate(sorted(groups, key=lambda s: (s != "built-in", s))):
        if i:
            parts.append(Text(""))
        group_tools = groups[source]
        table = _base_table(f"{source}  ({len(group_tools)})")
        table.expand = True  # let the ratio guidance column fill the width
        table.add_column("name", style=C_ACCENT, no_wrap=True)
        table.add_column("guidance", style=C_DIM, ratio=1)  # wraps to fill remaining width
        for t in group_tools:
            guidance = (getattr(t, "guidance", "") or "").strip().splitlines()[:1]
            text = re.sub(r"^[-•]\s*", "", guidance[0]) if guidance else ""  # drop the leading bullet
            table.add_row(t.name, text)
        parts.append(table)
    return Group(*parts)


def mcp_renderable(mcp_clients: dict) -> RenderableType:
    if not mcp_clients:
        return Text("no MCP servers connected", style=C_DIM)
    table = _base_table("MCP servers")
    table.add_column("name", style=C_ACCENT, no_wrap=True)
    table.add_column("url", style=C_DIM, overflow="ellipsis", no_wrap=True, max_width=48)
    table.add_column("tools", style=C_DIM, no_wrap=True, justify="right")
    for name, client in mcp_clients.items():
        try:
            n = str(len(client.list_tools()))
        except Exception:
            n = "?"
        url = getattr(client, "url", None) or "(stdio)"
        table.add_row(name, url, n)
    return table


def _base_table(title: str = "") -> Table:
    return Table(
        title=title or None,
        title_style="bold",
        title_justify="left",
        box=SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=C_DIM,
        padding=(0, 2, 0, 0),
    )


def tool_call_renderable(name: str, args_str: str) -> Text:
    line = Text("  ")
    line.append("▸ ", style=C_ACCENT)
    line.append(name, style="bold")
    if args_str:
        line.append(f"  {args_str}", style=C_DIM)
    return line


def tool_result_renderable(text: str, *, is_error: bool = False) -> Text:
    marker, style = ("✗ ", C_ERR) if is_error else ("↳ ", C_DIM)
    return Text("    " + marker + text, style=style)


def plan_renderable(plan: str) -> RenderableType:
    """An ExitPlanMode call's plan, rendered as markdown in a titled panel —
    shown at tool_start, before the approve/reject prompt appears below it."""
    body: RenderableType = Markdown(plan) if (plan or "").strip() else Text("(empty plan)", style=C_DIM)
    return Panel(body, title="proposed plan · awaiting approval", border_style=C_ACCENT, padding=(0, 1))


def print_plan(plan: str) -> None:
    console.print()
    console.print(plan_renderable(plan))


def ask_renderable(question: str, options: list | None = None) -> RenderableType:
    """An AskUser question, rendered in a titled panel — the model has paused
    mid-turn to ask the human. `options`, when given, are shown as a hint line;
    the answer is typed in either interface."""
    body = Text()
    body.append((question or "").strip() or "(no question)")
    if options:
        body.append("\n")
        body.append("options: " + "  ·  ".join(str(o) for o in options), style=C_DIM)
    return Panel(body, title="the agent needs your input", border_style=C_ACCENT, padding=(0, 1))


def status_bar(
    *, busy: bool, frame: str, label: str, spent: int, budget: int, status: str = "", steps: int | None = None
) -> RenderableType:
    """The fixed bottom status line: current state on the left, the token
    budget bar on the right. This is where the per-turn summary lives in the
    TUI — updated in place instead of appended after every message."""
    left = Text()
    if busy:
        left.append(f"{frame} {label}", style=C_ACCENT)
        left.append("   ctrl+c to stop", style=C_DIM)
    elif status:
        left.append("✓ " if status == "ok" else "• ", style=C_OK if status == "ok" else C_WARN)
        left.append(status, style=C_DIM)
        if steps is not None:
            left.append(f" · {steps} steps", style=C_DIM)
    else:
        left.append("ready", style=C_DIM)

    right = Text()
    if budget:
        width = 16
        frac = spent / budget
        over = spent > budget
        filled = min(width, round(frac * width))
        bar_style = C_ERR if (over or frac >= 0.9) else C_WARN if frac >= 0.7 else C_DIM
        right.append(f"{spent:,}/{budget:,}  ", style=C_DIM)
        right.append("█" * filled + "░" * (width - filled), style=bar_style)
        right.append(f"  {'over' if over else f'{frac * 100:.0f}%'}", style=C_ERR if over else C_DIM)
    else:
        right.append(f"{spent:,} tokens", style=C_DIM)

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(left, right)
    return grid


def user_line(msg: str) -> Text:
    t = Text()
    t.append(f"{PROMPT} ", style=C_ACCENT)
    t.append(msg)
    return t


def info_line(msg: str) -> Text:
    return Text(msg, style=C_DIM)


def turn_summary_renderable(result: Any, budget: int) -> Text:
    ok = result.status == "ok"
    sep = Text("  ·  ", style=C_DIM)
    t = Text()
    t.append("✓ " if ok else "• ", style=C_OK if ok else C_WARN)
    t.append(result.status, style=C_DIM)
    t.append_text(sep.copy())
    t.append(f"{result.steps} steps", style=C_DIM)
    t.append_text(sep.copy())
    t.append(f"{result.tokens_spent:,} tokens", style=C_DIM)
    if budget:
        width = 16
        frac = result.tokens_spent / budget
        over = result.tokens_spent > budget
        filled = min(width, round(frac * width))
        bar_style = C_ERR if (over or frac >= 0.9) else C_WARN if frac >= 0.7 else C_DIM
        t.append_text(sep.copy())
        t.append("█" * filled + "░" * (width - filled), style=bar_style)
        t.append(" ")
        t.append("over budget" if over else f"{frac * 100:.0f}%", style=C_ERR if over else C_DIM)
    return t


# ---------------------------------------------------------------------------
# printers — line-based CLI (wrap the builders)
# ---------------------------------------------------------------------------
def welcome(*, version: str, user: str, provider_label: str, model: str, backend: str, session_id: str) -> None:
    console.print(
        welcome_renderable(
            version=version, user=user, provider_label=provider_label, model=model, backend=backend, session_id=session_id
        )
    )


def mcp_connected(name: str, url: str, n_tools: int) -> None:
    console.print(mcp_connected_renderable(name, url, n_tools))


def mcp_failed(name: str, url: str, error: Exception) -> None:
    console.print(mcp_failed_renderable(name, url, error))


def session_info(session: Any) -> None:
    console.print(session_renderable(session))


def cli_help() -> None:
    console.print(cli_help_renderable())


def help_table() -> None:
    console.print(help_renderable())


def skills_table(skills: list) -> None:
    console.print(skills_renderable(skills))


def tools_table(tools: list) -> None:
    console.print(tools_renderable(tools))


def mcp_table(mcp_clients: dict) -> None:
    console.print(mcp_renderable(mcp_clients))


def tool_call(name: str, args_str: str) -> None:
    console.print(tool_call_renderable(name, args_str), no_wrap=True, overflow="ellipsis", crop=True)


def tool_result(text: str, *, is_error: bool = False) -> None:
    console.print(tool_result_renderable(text, is_error=is_error), no_wrap=True, overflow="ellipsis", crop=True)


def turn_summary(result: Any, budget: int) -> None:
    console.print("")
    console.print(turn_summary_renderable(result, budget))


def error(msg: str) -> None:
    console.print(f"[{C_ERR}]error:[/] {escape(msg)}")


def info(msg: str) -> None:
    console.print(f"[{C_DIM}]{escape(msg)}[/]")


def success(msg: str) -> None:
    console.print(f"[{C_OK}]{escape(msg)}[/]")


class AssistantStream:
    """Live-renders streamed assistant text as markdown for the line-based CLI.

    Buffers deltas and re-renders the whole span through Rich's Markdown on a
    throttled `Live`. `end()` freezes the final render; call it before any tool
    activity or the turn summary interrupts the prose."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._live: Live | None = None

    def feed(self, delta: str) -> None:
        if self._live is None:
            console.print()
            self._live = Live(console=console, refresh_per_second=12, vertical_overflow="visible")
            self._live.start()
        self._buf.append(delta)
        self._live.update(Markdown("".join(self._buf)))

    def end(self) -> None:
        if self._live is not None:
            self._live.update(Markdown("".join(self._buf)))
            self._live.stop()
            self._live = None
            self._buf = []


# ---------------------------------------------------------------------------
# RenderUI: whitelisted UI tree -> real terminal widgets
# ---------------------------------------------------------------------------
def ui_renderable(root: Any) -> RenderableType | None:
    """Build a RenderUI `root` node as terminal widgets, or None if the payload
    isn't a valid node (so the caller can fall back to raw call/result)."""
    if not isinstance(root, dict) or not isinstance(root.get("type"), str):
        return None
    return Padding(_node(root), (0, 0, 0, 2))


def render_ui(root: Any) -> bool:
    r = ui_renderable(root)
    if r is None:
        return False
    console.print()
    console.print(r)
    return True


def _node(n: Any) -> RenderableType:
    if not isinstance(n, dict):
        return Text(str(n))
    t = n.get("type", "")
    kids = lambda: [_node(c) for c in n.get("children", []) if isinstance(c, dict)]  # noqa: E731

    if t == "Stack":
        gap = n.get("gap", 1)
        children = kids()
        if gap and children:
            spaced: list[Any] = []
            for i, c in enumerate(children):
                if i:
                    spaced.extend([Text("")] * gap)
                spaced.append(c)
            return Group(*spaced)
        return Group(*children)
    if t == "Row":
        return Columns(kids(), padding=(0, 3), expand=False, equal=False)
    if t == "Grid":
        return Columns(kids(), padding=(0, 3), equal=True, column_first=False)
    if t == "Card":
        return Panel(
            Group(*kids()),
            title=escape(str(n["title"])) if n.get("title") else None,
            title_align="left",
            border_style=C_DIM,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
    if t == "Heading":
        return Text(str(n.get("text", "")), style="bold")
    if t == "Text":
        return Text(str(n.get("text", "")), style=C_DIM if n.get("muted") else "")
    if t == "Markdown":
        return Markdown(str(n.get("text", "")))
    if t == "Badge":
        return Text(f" {n.get('text', '')} ", style=f"reverse {_TONE.get(n.get('tone', 'neutral'), C_ACCENT)}")
    if t == "Stat":
        g = Text()
        g.append(str(n.get("label", "")) + "\n", style=C_DIM)
        g.append(str(n.get("value", "")), style="bold")
        if n.get("delta") not in (None, ""):
            g.append(f"  {n['delta']}", style=C_OK)
        return g
    if t == "Callout":
        tone = _TONE.get(n.get("tone", "neutral"), C_ACCENT)
        body = Text()
        if n.get("title"):
            body.append(str(n["title"]) + "\n", style="bold")
        body.append(str(n.get("text", "")))
        return Panel(body, border_style=tone, box=box.ROUNDED, padding=(0, 1), expand=False)
    if t == "Code":
        return Syntax(str(n.get("code", "")), str(n.get("lang", "text")), theme="ansi_dark", word_wrap=True)
    if t == "Divider":
        return Rule(style=C_DIM)
    if t == "Table":
        return _ui_table(n)
    if t == "Chart":
        return _ui_chart(n)
    if t == "Progress":
        return _ui_bar(n.get("value", 0), n.get("max", 100) or 100, n.get("label"), n.get("tone"))
    if t in ("Button", "Select", "Input", "Form"):
        return _ui_interactive(n)
    if n.get("children"):
        return Group(*kids())
    return Text(str(n.get("text", t)), style=C_DIM)


def _ui_table(n: dict) -> RenderableType:
    table = Table(box=SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold", padding=(0, 2, 0, 0))
    for col in n.get("columns", []):
        table.add_column(str(col))
    for row in n.get("rows", []):
        table.add_row(*[str(c) for c in row])
    return table


def _ui_chart(n: dict) -> RenderableType:
    series = [s for s in n.get("series", []) if isinstance(s, dict)]
    if not series:
        return Text("(empty chart)", style=C_DIM)
    values = [float(s.get("value", 0) or 0) for s in series]
    mx = max(values) or 1.0
    label_w = max((len(str(s.get("label", ""))) for s in series), default=0)
    body = Text()
    for s, v in zip(series, values, strict=True):
        fill = round(v / mx * 24)
        body.append(f"{str(s.get('label', '')):<{label_w}}  ", style=C_DIM)
        body.append("█" * max(fill, 1), style=C_ACCENT)
        body.append(f"  {s.get('value', '')}\n")
    if n.get("title"):
        return Group(Text(str(n["title"]), style="bold"), body)
    return body


def _ui_bar(value: Any, maximum: Any, label: Any, tone: Any) -> RenderableType:
    try:
        frac = max(0.0, min(1.0, float(value) / float(maximum)))
    except (TypeError, ZeroDivisionError, ValueError):
        frac = 0.0
    width = 24
    filled = round(frac * width)
    bar = Text()
    if label:
        bar.append(f"{label}  ", style=C_DIM)
    bar.append("█" * filled, style=_TONE.get(tone or "neutral", C_ACCENT))
    bar.append("░" * (width - filled), style=C_DIM)
    bar.append(f"  {value}/{maximum}", style=C_DIM)
    return bar


def _ui_interactive(n: dict) -> RenderableType:
    t = n.get("type")
    if t == "Button":
        return Text(f" {n.get('label', 'Button')} ", style=f"reverse {_TONE.get(n.get('tone', 'neutral'), C_ACCENT)}")
    if t == "Select":
        opts = ", ".join(str(o.get("label", o)) for o in n.get("options", []) if isinstance(o, dict))
        return Text(f"▾ select: {opts}", style=C_DIM)
    if t == "Input":
        return Text(f"▭ {n.get('placeholder', 'input')}", style=C_DIM)
    if t == "Form":
        inner = [_node(c) for c in n.get("children", []) if isinstance(c, dict)]
        submit = Text(f" {n.get('submitLabel', 'Submit')} ", style=f"reverse {C_ACCENT}")
        return Panel(Group(*inner, submit), border_style=C_DIM, box=box.ROUNDED, padding=(0, 1), expand=False)
    return Text(str(n), style=C_DIM)
