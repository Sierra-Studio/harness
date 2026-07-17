"""Command-line entrypoint.

    uv run harness init-db     # create schema in DATABASE_URL
    uv run harness chat        # interactive session (uses OpenRouter if configured)
    uv run harness serve [host] [port]
                               # HTTP server streaming turns as Server-Sent Events
                               # (host/port also via HARNESS_HTTP_HOST/HARNESS_HTTP_PORT)
    uv run harness add-skill <user_id> <name> <summary> [body]
                               # author a skill (body read from stdin if omitted)
    uv run harness list-skills <user_id>

(or without uv:  python -m harness.interfaces.cli init-db | chat | ...)

`chat` is a rich, slash-command-driven REPL — type `/help` once inside it for
the full command list (session info, skills, tools, live MCP connect, ...).

The CLI is an APPLICATION, not library-internal code — it is the one place
that opts into `Config.from_env()` and picks a provider from what's configured
(`detect_provider`) and a repository from `DATABASE_URL` (`build_repository`).
`Harness` itself never does either of those things on its own. It is also the
one place that loads a `.env` file (see `_load_dotenv`) — `uv run` does NOT do
this automatically, and nothing in `harness.settings` reads files, only
`os.environ`.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path
from typing import Literal

with contextlib.suppress(ImportError):
    # Loading readline transparently gives the plain REPL's input() line editing
    # and ↑/↓ history recall. Not available on some platforms — harmless if so.
    import readline  # noqa: F401

from .. import __version__
from ..core import Harness
from ..llm import detect_provider, provider_label
from ..persistence import build_repository
from ..settings import Config, mcp_http_servers
from ..tools import LocalSubprocessSandbox
from . import mcpstore, prefs, ui

_DOTENV_KEYS: set[str] = set()  # keys _load_dotenv introduced (vs. a real shell env var)


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a `.env` file, without overriding real env vars.

    Minimal, dependency-free `KEY=VALUE` parser: skips blank/comment lines,
    an optional leading `export `, strips a trailing ` # inline comment`
    (unless the value is quoted), and strips surrounding quotes. Silently
    does nothing if the file doesn't exist — `.env` is optional everywhere.

    Records which keys it actually introduced in `_DOTENV_KEYS`, so callers
    (see `prefs.apply_defaults`) can tell a `.env` default apart from a real
    env var the user actually set — a saved `/model`-style preference should
    still outrank the former, never the latter.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = value.strip("'\"")
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key not in os.environ:
            os.environ[key] = value
            _DOTENV_KEYS.add(key)


def init_db() -> int:
    cfg = Config.from_env()
    if not cfg.database_url:
        ui.error("DATABASE_URL is not set.")
        return 1
    import psycopg

    schema = (Path(__file__).resolve().parent.parent.parent / "schema.sql").read_text()
    with psycopg.connect(cfg.database_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(schema)
    ui.success(f"Schema applied to {cfg.database_url}.")
    return 0


def add_skill(argv: list[str]) -> int:
    """author a skill: add-skill <user_id> <name> <summary> [body]
    If body is omitted, it is read from stdin (so you can pipe or heredoc it)."""
    if len(argv) < 5:
        ui.error("usage: harness add-skill <user_id> <name> <summary> [body]")
        return 1
    user_id, name, summary = argv[2], argv[3], argv[4]
    body = argv[5] if len(argv) > 5 else sys.stdin.read()
    if not body.strip():
        ui.error("empty body (provide as arg or via stdin).")
        return 1
    cfg = Config.from_env()
    h = Harness(cfg, provider=detect_provider(cfg), repo=build_repository(cfg))
    uid = h.repo.get_or_create_user(user_id).id
    skill = h.skills.add(uid, name, summary, body.strip(), "authored")
    ui.success(f"Added skill '{skill.name}' for user '{user_id}' (id={skill.id}).")
    return 0


def list_skills(argv: list[str]) -> int:
    if len(argv) < 3:
        ui.error("usage: harness list-skills <user_id>")
        return 1
    cfg = Config.from_env()
    h = Harness(cfg, provider=detect_provider(cfg), repo=build_repository(cfg))
    uid = h.repo.get_or_create_user(argv[2]).id
    ui.skills_table(h.skills.list(uid))
    return 0


def _skills_add(h: Harness, session, args: list[str]) -> None:
    if len(args) < 2:
        ui.error("usage: /skills add <name> <summary> [body...]")
        return
    name, summary, *rest = args
    body = " ".join(rest) if rest else summary
    skill = h.skills.add(session.user_id, name, summary, body, "authored")
    ui.success(f"added skill '{skill.name}' (id={skill.id})")


def _mcp_http(h: Harness, args: list[str]) -> None:
    if not args:
        ui.error("usage: /mcp http <url> [name] [--direct]")
        return
    direct = "--direct" in args
    args = [a for a in args if a != "--direct"]
    url, name = args[0], (args[1] if len(args) > 1 else "")
    expose: Literal["index", "direct"] = "direct" if direct else "index"
    try:
        client = h.add_mcp_http(url, name, expose=expose)
        mcpstore.save(client.name, url, expose)  # persist so it reconnects next launch
        ui.mcp_connected(client.name, url, len(client.list_tools()))
    except Exception as e:
        ui.mcp_failed(name or url, url, e)


def _mcp_stdio(h: Harness, args: list[str]) -> None:
    if len(args) < 2:
        ui.error("usage: /mcp stdio <name> <cmd...> [--direct]")
        return
    direct = "--direct" in args
    args = [a for a in args if a != "--direct"]
    name, command = args[0], args[1:]
    try:
        client = h.add_mcp_stdio(command, name, expose="direct" if direct else "index")
        ui.mcp_connected(client.name, " ".join(command), len(client.list_tools()))
    except Exception as e:
        ui.mcp_failed(name, " ".join(command), e)


def _copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard write for the line-based CLI via the OS tool."""
    import shutil
    import subprocess

    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True)
                return True
            except Exception:
                return False
    return False


def _save_transcript(h: Harness, session, path: str = "") -> str:
    dest = Path(path) if path else Path(f"harness-{str(session.id)[:8]}.md")
    dest.write_text(ui.transcript_markdown(h.repo.active_turns(session.id)))
    return str(dest)


def _resume(h: Harness, external_id: str, arg: str):
    """Resolve and reopen a session; returns (session, message) or (None, error)."""
    uid = h.repo.get_or_create_user(external_id).id
    summ = ui.resolve_resume(h.repo, uid, arg)
    if summ is None:
        return None, f"no matching session for {arg!r}" if arg else "no sessions to resume"
    session = h.start_session(external_id, session_id=summ.id)
    h.repo.set_session_status(session.id, "open")
    session.status = "open"
    subject = f" · {summ.subject}" if summ.subject else ""
    return session, f"resumed {summ.id[:8]} · {summ.turns} turns{subject}"


def _run_command(h: Harness, session, external_id: str, line: str, run, last):
    """Handle a `/`-prefixed slash command. `run(msg)` executes a turn (for
    /retry); `last` is a dict with the last user message and assistant answer.
    Returns the (possibly new) session."""
    cmd, args, raw = ui.parse_command(line)

    if cmd == "/help":
        ui.help_table()
    elif cmd == "/clear":
        ui.console.clear()
    elif cmd == "/session":
        # session.tokens_spent isn't mutated in place by run_turn_stream — only
        # the repo row is updated — so re-fetch for an accurate token count.
        ui.session_info(h.repo.get_session(session.id))
    elif cmd == "/skills":
        if args[:1] == ["add"]:
            _skills_add(h, session, args[1:])
        else:
            ui.skills_table(h.skills.list(session.user_id))
    elif cmd == "/tools":
        ui.tools_table(h.tools.active_tools())
    elif cmd == "/sessions":
        uid = h.repo.get_or_create_user(external_id).id
        ui.console.print(ui.sessions_renderable(h.repo.list_sessions(uid), active_id=session.id))
    elif cmd == "/resume":
        new_session, msg = _resume(h, external_id, args[0] if args else "")
        (ui.success if new_session else ui.error)(msg)
        session = new_session or session
    elif cmd == "/retry":
        if last.get("user"):
            run(last["user"])
        else:
            ui.error("nothing to retry yet")
    elif cmd == "/copy":
        answer = last.get("answer", "")
        if not answer:
            ui.error("no answer to copy yet")
        elif _copy_to_clipboard(answer):
            ui.success(f"copied {len(answer)} chars to the clipboard")
        else:
            ui.error("no clipboard tool found (pbcopy/xclip/wl-copy/clip)")
    elif cmd == "/save":
        dest = _save_transcript(h, session, args[0] if args else "")
        ui.success(f"saved transcript to {dest}")
    elif cmd == "/persona":
        if not args or ui.is_query(args):
            current = prefs.load()["persona"]
            ui.info(f"persona: {current}" if current else "persona: (default identity)")
        elif args == ["clear"]:
            h.set_persona()
            prefs.save(persona="", system_prompt="")
            ui.success("persona reset to the default identity")
        else:
            h.set_persona(persona=raw)
            prefs.save(persona=raw, system_prompt="")
            ui.success("persona updated for this session and saved as the default")
    elif cmd == "/system-prompt":
        if not args or ui.is_query(args):
            current = prefs.load()["system_prompt"]
            ui.info(f"system prompt: {current}" if current else "system prompt: (none — using persona layering)")
        elif args == ["clear"]:
            h.set_persona()
            prefs.save(persona="", system_prompt="")
            ui.success("system-prompt override cleared")
        else:
            h.set_persona(system_prompt=raw)
            prefs.save(system_prompt=raw, persona="")
            ui.success("system-prompt override applied for this session and saved as the default")
    elif cmd == "/model":
        if not args or ui.is_query(args):
            ui.info(f"model: {session.model}  (default: {prefs.load()['model'] or h.cfg.provider.model})")
        else:
            model = args[0]
            available = h.provider.available_models()
            if available is not None and model not in available:
                import difflib

                near = difflib.get_close_matches(model, available, n=3, cutoff=0.4)
                near += [m for m in available if model.lower() in m.lower() and m not in near]
                hint = f"  did you mean: {', '.join(near[:3])}?" if near else ""
                ui.error(f"unknown model '{model}' — not saved.{hint}")
            else:
                h.set_session_model(session, model)
                prefs.save(model=model)
                ui.success(f"model set to '{model}' for this session and saved as the default")
    elif cmd == "/budget":
        if not args or ui.is_query(args):
            ui.info(f"budget: {session.token_budget or 'unlimited'}")
        else:
            parsed = ui.parse_budget(args[0])
            if parsed is None:
                ui.error("usage: /budget <n>|unlimited  (e.g. /budget 200000, /budget unlimited)")
            else:
                h.set_session_budget(session, parsed)
                prefs.save(token_budget=parsed)
                ui.success(f"budget set to {parsed or 'unlimited'} for this session and saved as the default")
    elif cmd == "/theme":
        if not args or ui.is_query(args):
            ui.info(f"theme: {prefs.load()['theme'] or '(default)'} — affects the TUI only")
        else:
            prefs.save(theme=args[0])
            ui.success(f"theme '{args[0]}' saved — takes effect next time you open the TUI")
    elif cmd == "/mcp":
        if not args:
            ui.mcp_table(h.tools.mcp_clients)
        elif args[0] == "http":
            _mcp_http(h, args[1:])
        elif args[0] == "stdio":
            _mcp_stdio(h, args[1:])
        elif args[0] == "remove" and len(args) > 1:
            ui.success(f"removed saved server {args[1]}") if mcpstore.remove(args[1]) else ui.error(
                f"no saved server {args[1]!r}"
            )
        else:
            ui.error("usage: /mcp [http <url> [name]] | [stdio <name> <cmd...>] | [remove <name>] [--direct]")
    elif cmd in ("/auto", "/manual", "/plan"):
        mode = h.set_permission_mode(cmd[1:])
        prefs.save(permission_mode=mode)
        ui.success(f"permission mode: {mode}")
    elif cmd == "/mode":
        ui.info(f"permission mode: {h.permissions.mode}  (/auto · /plan · /manual)")
    elif cmd == "/new":
        h.close_session(session)
        session = h.start_session(external_id)
        ui.success(f"started new session {session.id}")
    else:
        ui.error(f"unknown command {cmd!r} — try /help")
    return session


def _default_user() -> str:
    """Local single-user default so `harness chat` never blocks on a prompt.
    Override with a positional arg (`harness chat alice`) or `HARNESS_USER`."""
    import getpass

    try:
        return getpass.getuser() or "local"
    except Exception:
        return "local"


class _TurnView:
    """Drives the live rendering of one turn: a spinner while the model thinks
    or a tool runs, markdown as it streams, and clean tool activity in between.

    Only one Rich `Live` may own the console at a time, so the spinner and the
    markdown stream are strictly interleaved — one is always stopped before the
    other starts."""

    def __init__(self, token_budget: int) -> None:
        self._budget = token_budget
        self._status = ui.console.status("[dim]thinking…[/]", spinner="dots")
        self._status_on = False
        self._stream = ui.AssistantStream()
        self._rendered_ui: set[str] = set()
        self._answer: list[str] = []
        self.answer = ""  # the turn's assistant text, for /copy

    def _spin(self, label: str) -> None:
        self._unspin()
        self._status.update(f"[dim]{label}[/]")
        self._status.start()
        self._status_on = True

    def _unspin(self) -> None:
        if self._status_on:
            self._status.stop()
            self._status_on = False

    def begin(self) -> None:
        self._spin("thinking…")

    def handle(self, ev) -> None:
        if ev.kind == "text":
            self._unspin()
            self._answer.append(ev.text)
            self._stream.feed(ev.text)
        elif ev.kind == "tool_start":
            self._unspin()
            self._stream.end()
            if ev.name == "RenderUI" and ui.render_ui(ev.args.get("root")):
                self._rendered_ui.add(ev.call_id)
            elif ev.name == "ExitPlanMode":
                # Not added to _rendered_ui — the normal tool_result line
                # below still prints the approve/reject outcome.
                ui.print_plan(ev.args.get("plan", ""))
            else:
                ui.tool_call(ui.tool_display_name(ev.name, ev.args), ui.tool_args_str(ev.name, ev.args))
            self._spin(f"running {ui.tool_display_name(ev.name, ev.args)}…")
        elif ev.kind == "tool_result":
            self._unspin()
            if ev.call_id not in self._rendered_ui:  # RenderUI already drew its widget
                ui.tool_result(ui.summarize_result(ev.content), is_error=ui.is_error_result(ev.content))
            self._spin("thinking…")
        elif ev.kind == "final":
            self.close()
            self.answer = "".join(self._answer)
            ui.turn_summary(ev.result, self._budget)

    def close(self) -> None:
        self._unspin()
        self._stream.end()


def _connect_mcp(h: Harness) -> list[tuple]:
    """Connect HTTP MCP servers from the env (MCP_HTTP_SERVERS) plus any saved
    from a prior `/mcp http` (mcpstore). Returns ('ok'|'err', name, url,
    n_tools|error_str) tuples so either interface can render the outcome."""
    results: list[tuple] = []
    seen: set[str] = set()

    def connect(
        name: str,
        url: str,
        headers=None,
        oauth=None,
        expose: Literal["index", "direct"] = "index",
    ) -> None:
        key = name or url
        if key in seen:
            return
        seen.add(key)
        try:
            client = h.add_mcp_http(url, name, headers, oauth=oauth, expose=expose)
            results.append(("ok", client.name, url, len(client.list_tools())))
        except Exception as e:
            results.append(("err", name or url, url, str(e)))

    for srv in mcp_http_servers():
        connect(srv["name"], srv["url"], srv["headers"], srv.get("oauth"))
    for srv in mcpstore.load():
        # sanitize at the boundary: the store is a hand-editable JSON file, so
        # anything that isn't exactly "direct" falls back to the safe default
        connect(
            srv.get("name", ""),
            srv["url"],
            expose="direct" if srv.get("expose") == "direct" else "index",
        )
    return results


def chat(argv: list[str]) -> int:
    """Dispatch to the full-screen Textual TUI in an interactive terminal, else
    the line-based REPL (piped/non-tty, `--plain`, or if Textual won't import)."""
    cfg = prefs.apply_defaults(Config.from_env(), from_dotenv=_DOTENV_KEYS)
    saved = prefs.load()
    h = Harness(
        cfg,
        provider=detect_provider(cfg),
        repo=build_repository(cfg),
        echo=False,
        persona=saved["persona"],
        system_prompt=saved["system_prompt"],
        # Interactive chat runs Bash in the directory you launched from, so the
        # agent works on your actual project (like Claude Code) instead of an
        # isolated empty tempdir. The workspace is never deleted on session end.
        sandbox=LocalSubprocessSandbox(max_output=cfg.bash.max_output, workspace=os.getcwd()),
    )
    external_id = (
        (argv[2] if len(argv) > 2 and not argv[2].startswith("-") else "").strip()
        or os.environ.get("HARNESS_USER", "").strip()
        or _default_user()
    )
    is_tty = sys.stdout.isatty() and sys.stdin.isatty()
    plain = "--plain" in argv or not is_tty
    # --inline: the Claude-Code-style front-end (native scrollback/selection/copy,
    # bordered box). Opt-in and TTY-only; ignored when piped or --plain.
    inline = "--inline" in argv and is_tty and "--plain" not in argv

    run_tui = None
    if not plain and not inline:
        try:
            from .tui import run_tui
        except Exception:
            run_tui = None

    try:
        if inline:
            return _chat_inline(h, cfg, external_id)
        if run_tui is not None:
            mcp_lines = _connect_mcp(h)
            session = h.start_session(external_id)
            return run_tui(h, session, external_id, provider_label(cfg), cfg.provider.model, mcp_lines)
        return _chat_plain(h, cfg, external_id)
    finally:
        for client in h.tools.mcp_clients.values():
            with contextlib.suppress(Exception):
                client.stop()


def _install_line_prompts(h: Harness) -> None:
    """Wire the blocking stdin prompts the synchronous turn loop uses for
    manual/plan-mode approvals and mid-turn AskUser. Shared by both line-based
    front-ends (`_chat_plain` and `_chat_inline`) since neither has the TUI's
    inline widgets — the turn runs on this thread and simply blocks on input."""

    def ask_permission(name: str, args: dict) -> str:
        from ..core import ALLOW, ALWAYS, DENY

        if name == "ExitPlanMode":
            ui.console.print(
                f"\n[{ui.C_ACCENT}]permission[/] approve this plan and start implementing?  "
                "[dim][y]approve [n]reject[/]"
            )
        else:
            label = ui.tool_display_name(name, args)
            ui.console.print(f"\n[{ui.C_ACCENT}]permission[/] {label}  [dim][y]allow [a]always [n]deny[/]")
        try:
            answer = ui.console.input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return DENY
        decision = {"y": ALLOW, "a": ALWAYS, "": ALLOW}.get(answer, DENY)
        if name == "ExitPlanMode" and decision in (ALLOW, ALWAYS):
            # NOT h.set_permission_mode — that would deregister ExitPlanMode
            # before ToolRegistry.dispatch looks it up for this same call.
            # See sync_plan_mode_tool's docstring / the TUI's _ask_permission.
            h.permissions.set_mode("manual")
            prefs.save(permission_mode="manual")
            ui.success("plan approved — now in manual mode; each tool call needs approval")
        return decision

    h.permissions.asker = ask_permission

    def ask_user(question: str, meta: dict) -> str:
        ui.console.print()
        ui.console.print(ui.ask_renderable(question, meta.get("options")))
        try:
            return ui.console.input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    h.tools.prompter = ask_user


def _chat_plain(h: Harness, cfg: Config, external_id: str) -> int:
    backend = "Postgres" if cfg.database_url else "in-memory"
    mcp_lines = _connect_mcp(h)
    session = h.start_session(external_id)
    ui.welcome(
        version=__version__,
        user=external_id,
        provider_label=provider_label(cfg),
        model=cfg.provider.model,
        backend=backend,
        session_id=session.id,
    )
    for kind, name, url, extra in mcp_lines:
        if kind == "ok":
            ui.mcp_connected(name, url, extra)
        else:
            ui.mcp_failed(name, url, Exception(extra))
    state = {"session": session, "last": {"user": "", "answer": ""}}
    _install_line_prompts(h)

    def run(message: str) -> None:
        state["last"]["user"] = message
        view = _TurnView(state["session"].token_budget)
        try:
            view.begin()
            for ev in h.run_turn_stream(state["session"], message):
                view.handle(ev)
        finally:
            view.close()
        state["last"]["answer"] = view.answer

    try:
        while True:
            msg = ui.console.input(f"\n[{ui.C_ACCENT}]{ui.PROMPT}[/] ").strip()
            if not msg:
                continue
            if msg in {"exit", "quit", "/exit", "/quit"}:
                break
            if msg.startswith("/"):
                ui.console.print()
                state["session"] = _run_command(h, state["session"], external_id, msg, run, state["last"])
                continue
            run(msg)
    except (EOFError, KeyboardInterrupt):
        pass
    created = h.close_session(state["session"])
    if created:
        ui.info(f"induced skills: {created}")
    ui.info("session closed.")
    return 0


def _chat_inline(h: Harness, cfg: Config, external_id: str) -> int:
    """The Claude-Code-style inline front-end: identical turn/command/streaming
    engine as `_chat_plain`, but input is read through a bordered box pinned at
    the bottom (see inputbox.read_boxed). It never enters the alternate screen
    or captures the mouse, so the terminal keeps native scrollback, wheel-scroll,
    text selection and copy — the whole point of this mode. The box only exists
    while awaiting input; on submit it's erased, the message is committed to
    scrollback, and the turn's output streams in above the next box."""
    from . import inputbox

    backend = "Postgres" if cfg.database_url else "in-memory"
    mcp_lines = _connect_mcp(h)
    session = h.start_session(external_id)
    ui.welcome(
        version=__version__,
        user=external_id,
        provider_label=provider_label(cfg),
        model=cfg.provider.model,
        backend=backend,
        session_id=session.id,
    )
    for kind, name, url, extra in mcp_lines:
        if kind == "ok":
            ui.mcp_connected(name, url, extra)
        else:
            ui.mcp_failed(name, url, Exception(extra))
    state = {"session": session, "last": {"user": "", "answer": ""}}
    _install_line_prompts(h)
    history: list[str] = []

    def run(message: str) -> None:
        state["last"]["user"] = message
        view = _TurnView(state["session"].token_budget)
        try:
            view.begin()
            for ev in h.run_turn_stream(state["session"], message):
                view.handle(ev)
        finally:
            view.close()
        state["last"]["answer"] = view.answer

    try:
        while True:
            ui.console.print()  # a breath of space between the box and prior output
            try:
                line = inputbox.read_boxed(ui.console, prompt=f"{ui.PROMPT} ", history=history)
            except KeyboardInterrupt:
                break
            if line is None:  # Ctrl-D on an empty box
                break
            msg = line.strip()
            if not msg:
                continue
            history.append(msg)
            # The box was erased on submit — echo the message into scrollback so
            # the conversation reads top-to-bottom like any shell session.
            ui.console.print(f"[{ui.C_ACCENT}]{ui.PROMPT}[/] {msg}")
            if msg in {"exit", "quit", "/exit", "/quit"}:
                break
            if msg.startswith("/"):
                ui.console.print()
                state["session"] = _run_command(h, state["session"], external_id, msg, run, state["last"])
                continue
            run(msg)
    except EOFError:
        pass
    created = h.close_session(state["session"])
    if created:
        ui.info(f"induced skills: {created}")
    ui.info("session closed.")
    return 0


def serve(argv: list[str]) -> int:
    from .server import serve as run_server

    host = argv[2] if len(argv) > 2 else os.environ.get("HARNESS_HTTP_HOST", "127.0.0.1")
    port = int(argv[3] if len(argv) > 3 else os.environ.get("HARNESS_HTTP_PORT", "8800"))
    return run_server(host, port)


def main(argv: list[str]) -> int:
    # Config precedence (highest first): real shell env var > a `.env` in the
    # current folder (project-local override) > a global `~/.harness/.env`. This
    # lets `harness` run from ANY directory once you drop your keys in the global
    # file, while a project can still override per-folder — `_load_dotenv` only
    # fills keys not already set, so loading cwd before global gives cwd priority.
    _load_dotenv()
    _load_dotenv(str(Path.home() / ".harness" / ".env"))
    cmd = argv[1] if len(argv) > 1 else "chat"
    if cmd in ("--version", "-V", "version"):
        print(f"harness {__version__}")
        return 0
    if cmd in ("--help", "-h", "help"):
        ui.cli_help()
        return 0
    if cmd == "init-db":
        return init_db()
    if cmd == "chat":
        return chat(argv)
    if cmd == "serve":
        return serve(argv)
    if cmd == "add-skill":
        return add_skill(argv)
    if cmd == "list-skills":
        return list_skills(argv)
    ui.cli_help()
    return 1


def entry() -> int:
    """Console-script entry point (`harness ...`)."""
    return main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
