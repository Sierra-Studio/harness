"""Full-screen Textual chat app for `harness chat` in an interactive terminal.

Layout: a header, a scrollable message history, a status bar (animated spinner
+ live token budget), and a docked input. Each turn runs in a worker *thread*
(`run_turn_stream` is a blocking synchronous generator), posting every event
back to the UI thread via `call_from_thread`, so the interface stays responsive
— the spinner keeps animating and the input never freezes while the model works.

All visible widgets are built by the shared `ui.*_renderable()` functions, so
the TUI and the line-based CLI render identically. Falls back to the line-based
REPL (see `cli.py`) when stdout/stdin aren't a TTY.
"""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.suggester import SuggestFromList
from textual.widgets import Header, Input, Static

from . import mcpstore, prefs, ui

# slash commands offered as ghost-text autocomplete (ordered by how completions resolve)
_SLASH_NAMES = [
    "/help", "/session", "/sessions", "/skills", "/skills add", "/tools", "/mcp",
    "/resume", "/retry", "/copy", "/save", "/persona", "/system-prompt", "/model",
    "/budget", "/theme", "/clear", "/new", "/exit", "/quit",
]


class PromptInput(Input):
    """Single-line input with shell-style ↑/↓ command history.

    A single-line `Input` doesn't use up/down for cursor movement, so we bind
    them to walk the submitted-message history. Navigating from a half-typed
    line stashes it as a draft, restored when you arrow back past the newest
    entry."""

    BINDINGS = [
        Binding("up", "history_prev", show=False),
        Binding("down", "history_next", show=False),
        Binding("tab", "complete", show=False, priority=True),
    ]

    def action_complete(self) -> None:
        """Tab accepts the ghost-text suggestion instead of moving focus away."""
        suggestion = self._suggestion
        if suggestion and len(suggestion) > len(self.value):
            self.value = suggestion
            self.cursor_position = len(self.value)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, suggester=SuggestFromList(_SLASH_NAMES, case_sensitive=False), **kwargs)
        self.history: list[str] = []
        self._idx: int | None = None
        self._draft = ""

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._idx = None
        self._draft = ""

    def action_history_prev(self) -> None:
        if not self.history:
            return
        if self._idx is None:
            self._draft = self.value
            self._idx = len(self.history) - 1
        elif self._idx > 0:
            self._idx -= 1
        self.value = self.history[self._idx]
        self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        if self._idx is None:
            return
        if self._idx < len(self.history) - 1:
            self._idx += 1
            self.value = self.history[self._idx]
        else:
            self._idx = None
            self.value = self._draft
        self.cursor_position = len(self.value)


class HarnessApp(App):
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #log > * { width: 1fr; height: auto; }
    #status { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
    #prompt { dock: bottom; height: 5; margin-bottom: 1; border: none;
              border-top: tall $warning; border-bottom: tall $warning; padding: 1 1; }
    #prompt:focus { border: none; border-top: tall $warning; border-bottom: tall $warning; }
    """

    BINDINGS = [
        ("ctrl+c", "interrupt", "stop / quit"),
        ("escape", "interrupt", "stop"),
        ("ctrl+d", "quit", "quit"),
        ("ctrl+l", "clear_log", "clear"),
    ]

    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self,
        harness: Any,
        session: Any,
        external_id: str,
        provider_label: str,
        model: str,
        mcp_lines: list[tuple],
    ) -> None:
        super().__init__()
        self.harness = harness
        self.session = session
        self.external_id = external_id
        self.provider_label = provider_label
        self.model = model
        self.mcp_lines = mcp_lines
        self._md_buf: list[str] = []
        self._md_widget: Static | None = None
        self._rendered_ui: set[str] = set()
        self._busy = False
        self._label = ""
        self._spin_i = 0
        self._spent = 0
        self._last_status = ""  # last turn's status/steps, shown in the fixed status bar
        self._last_steps: int | None = None
        self._answer: list[str] = []  # current turn's assistant text
        self._last_user = ""  # for /retry
        self._last_answer = ""  # for /copy
        self._cancel = False  # set by ctrl+c to stop the running turn
        self._pending: list[str] = []  # buffered lines of a backslash-continued message
        saved_theme = prefs.load()["theme"]
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme

    def watch_theme(self, theme_name: str) -> None:
        """Persist the theme regardless of how it was changed — the ctrl+p
        command palette's built-in theme picker, or /theme."""
        prefs.save(theme=theme_name)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="log")
        yield Static("", id="status")
        yield PromptInput(placeholder="Message…   (/help · ↑ history · ctrl+c stop)", id="prompt")

    def on_mount(self) -> None:
        self.title = "harness"
        self.sub_title = self.external_id
        self._write(ui.welcome_renderable(
            version=_version(),
            user=self.external_id,
            provider_label=self.provider_label,
            model=self.model,
            backend=_backend(self.harness),
            session_id=self.session.id,
        ))
        for kind, name, url, extra in self.mcp_lines:
            if kind == "ok":
                self._write(ui.mcp_connected_renderable(name, url, extra))
            else:
                self._write(ui.mcp_failed_renderable(name, url, Exception(extra)))
        self.set_interval(0.1, self._tick)
        self.query_one("#prompt", Input).focus()
        self._refresh_status()

    # -- rendering helpers --------------------------------------------------
    def _write(self, renderable: Any) -> Static:
        log = self.query_one("#log", VerticalScroll)
        widget = Static(renderable)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    def _end_md(self) -> None:
        self._md_widget = None
        self._md_buf = []

    def _tick(self) -> None:
        if self._busy:
            self._spin_i = (self._spin_i + 1) % len(self.SPINNER)
            self._refresh_status()

    def _refresh_status(self) -> None:
        budget = self.session.token_budget
        self.query_one("#status", Static).update(
            ui.status_bar(
                busy=self._busy,
                frame=self.SPINNER[self._spin_i],
                label=self._label,
                spent=self._spent,
                budget=budget,
                status=self._last_status,
                steps=self._last_steps,
            )
        )

    # -- input --------------------------------------------------------------
    @on(Input.Submitted, "#prompt")
    def _submit(self, event: Input.Submitted) -> None:
        raw = event.value
        event.input.value = ""
        # multi-line: a trailing backslash buffers the line and waits for more
        if raw.endswith("\\"):
            self._pending.append(raw[:-1])
            event.input.placeholder = "… continuation (send a line without a trailing \\)"
            return
        if self._pending:
            raw = "\n".join([*self._pending, raw])
            self._pending = []
            event.input.placeholder = "Message…   (/help · ↑ history · ctrl+c stop)"
        msg = raw.strip()
        if not msg:
            return
        self.query_one("#prompt", PromptInput).remember(msg)
        if msg in {"exit", "quit", "/exit", "/quit"}:
            self.exit()
            return
        self._write(ui.user_line(msg))
        if msg.startswith("/"):
            self._slash(msg)
            return
        self._begin_turn(msg)

    def _begin_turn(self, msg: str) -> None:
        self._busy = True
        self._cancel = False
        self._last_user = msg
        self._answer = []
        self._label = "thinking…"
        self.query_one("#prompt", Input).disabled = True
        self._refresh_status()
        self._run_turn(msg)

    @work(thread=True, exclusive=True, group="turn")
    def _run_turn(self, msg: str) -> None:
        try:
            for ev in self.harness.run_turn_stream(self.session, msg):
                if self._cancel:  # ctrl+c between events — stop feeding the UI
                    break
                self.call_from_thread(self._handle, ev)
        except Exception as e:  # keep the app alive; surface the failure
            self.call_from_thread(self._write, ui.tool_result_renderable(str(e), is_error=True))
        finally:
            self.call_from_thread(self._end_turn)

    def _handle(self, ev: Any) -> None:
        if ev.kind == "text":
            self._md_buf.append(ev.text)
            self._answer.append(ev.text)
            if self._md_widget is None:
                self._md_widget = self._write(Markdown("".join(self._md_buf)))
            else:
                self._md_widget.update(Markdown("".join(self._md_buf)))
                self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        elif ev.kind == "tool_start":
            self._end_md()
            name = ui.tool_display_name(ev.name, ev.args)
            widget = ui.ui_renderable(ev.args.get("root")) if ev.name == "RenderUI" else None
            if widget is not None:
                self._write(widget)
                self._rendered_ui.add(ev.call_id)
            else:
                self._write(ui.tool_call_renderable(name, ui.tool_args_str(ev.name, ev.args)))
            self._label = f"running {name}…"
            self._refresh_status()
        elif ev.kind == "tool_result":
            if ev.call_id not in self._rendered_ui:
                self._write(ui.tool_result_renderable(ui.summarize_result(ev.content), is_error=ui.is_error_result(ev.content)))
            self._label = "thinking…"
            self._refresh_status()
        elif ev.kind == "final":
            self._end_md()
            self._spent = ev.result.tokens_spent
            self._last_status = ev.result.status
            self._last_steps = ev.result.steps
            # the per-turn summary lives in the fixed status bar, not the log

    def _end_turn(self) -> None:
        self._end_md()
        self._last_answer = "".join(self._answer)
        if self._cancel:
            self._write(ui.info_line("stopped."))
        self._busy = False
        self._cancel = False
        self._label = ""
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        inp.focus()
        self._refresh_status()

    def action_interrupt(self) -> None:
        """Ctrl+C / Esc: stop the running turn if busy, else quit the app."""
        if self._busy:
            self._cancel = True
            self._label = "stopping…"
            self._refresh_status()
        else:
            self.exit()

    # -- slash commands -----------------------------------------------------
    def _slash(self, line: str) -> None:
        cmd, args, raw = ui.parse_command(line)
        if cmd == "/help":
            self._write(ui.help_renderable())
        elif cmd == "/clear":
            self.action_clear_log()
        elif cmd == "/session":
            self._write(ui.session_renderable(self.harness.repo.get_session(self.session.id)))
        elif cmd == "/skills":
            if args[:1] == ["add"]:
                self._skills_add(args[1:])
            else:
                self._write(ui.skills_renderable(self.harness.skills.list(self.session.user_id)))
        elif cmd == "/tools":
            self._write(ui.tools_renderable(self.harness.tools.active_tools()))
        elif cmd == "/sessions":
            uid = self.harness.repo.get_or_create_user(self.external_id).id
            self._write(ui.sessions_renderable(self.harness.repo.list_sessions(uid), active_id=self.session.id))
        elif cmd == "/resume":
            self._resume(args[0] if args else "")
        elif cmd == "/retry":
            if self._last_user:
                self._write(ui.user_line(self._last_user))
                self._begin_turn(self._last_user)
            else:
                self._write(ui.info_line("nothing to retry yet"))
        elif cmd == "/copy":
            self._copy()
        elif cmd == "/save":
            self._save(args[0] if args else "")
        elif cmd == "/persona":
            self._persona(args, raw)
        elif cmd == "/system-prompt":
            self._system_prompt(args, raw)
        elif cmd == "/model":
            self._model(args)
        elif cmd == "/budget":
            self._budget(args)
        elif cmd == "/theme":
            self._theme(args)
        elif cmd == "/mcp":
            self._mcp(args)
        elif cmd == "/new":
            self.harness.close_session(self.session)
            self.session = self.harness.start_session(self.external_id)
            self._spent = 0
            self._write(ui.info_line(f"started new session {self.session.id}"))
            self._refresh_status()
        else:
            self._write(ui.tool_result_renderable(f"unknown command {cmd} — try /help", is_error=True))

    def _resume(self, arg: str) -> None:
        uid = self.harness.repo.get_or_create_user(self.external_id).id
        summ = ui.resolve_resume(self.harness.repo, uid, arg)
        if summ is None:
            self._write(ui.tool_result_renderable(f"no matching session for {arg!r}" if arg else "no sessions to resume", is_error=True))
            return
        self.session = self.harness.start_session(self.external_id, session_id=summ.id)
        self.harness.repo.set_session_status(self.session.id, "open")
        self.session.status = "open"
        self._spent = self.session.tokens_spent
        subject = f" · {summ.subject}" if summ.subject else ""
        self._write(ui.info_line(f"resumed {summ.id[:8]} · {summ.turns} turns{subject}"))
        self._refresh_status()

    def _skills_add(self, args: list[str]) -> None:
        if len(args) < 2:
            self._write(ui.tool_result_renderable("usage: /skills add <name> <summary> [body...]", is_error=True))
            return
        name, summary, *rest = args
        body = " ".join(rest) if rest else summary
        skill = self.harness.skills.add(self.session.user_id, name, summary, body, "authored")
        self._write(ui.info_line(f"added skill '{skill.name}' (id={skill.id})"))

    def _persona(self, args: list[str], raw: str) -> None:
        if not args or ui.is_query(args):
            current = prefs.load()["persona"]
            self._write(ui.info_line(f"persona: {current}" if current else "persona: (default identity)"))
        elif args == ["clear"]:
            self.harness.set_persona()
            prefs.save(persona="", system_prompt="")
            self._write(ui.info_line("persona reset to the default identity"))
        else:
            self.harness.set_persona(persona=raw)
            prefs.save(persona=raw, system_prompt="")
            self._write(ui.info_line("persona updated for this session and saved as the default"))

    def _system_prompt(self, args: list[str], raw: str) -> None:
        if not args or ui.is_query(args):
            current = prefs.load()["system_prompt"]
            self._write(
                ui.info_line(f"system prompt: {current}" if current else "system prompt: (none — using persona layering)")
            )
        elif args == ["clear"]:
            self.harness.set_persona()
            prefs.save(persona="", system_prompt="")
            self._write(ui.info_line("system-prompt override cleared"))
        else:
            self.harness.set_persona(system_prompt=raw)
            prefs.save(system_prompt=raw, persona="")
            self._write(ui.info_line("system-prompt override applied for this session and saved as the default"))

    def _model(self, args: list[str]) -> None:
        if not args or ui.is_query(args):
            default = prefs.load()["model"] or self.harness.cfg.provider.model
            self._write(ui.info_line(f"model: {self.session.model}  (default: {default})"))
        else:
            self.harness.set_session_model(self.session, args[0])
            prefs.save(model=args[0])
            self._write(ui.info_line(f"model set to '{args[0]}' for this session and saved as the default"))
            self._refresh_status()

    def _budget(self, args: list[str]) -> None:
        if not args or ui.is_query(args):
            self._write(ui.info_line(f"budget: {self.session.token_budget or 'unlimited'}"))
        else:
            parsed = ui.parse_budget(args[0])
            if parsed is None:
                self._write(
                    ui.tool_result_renderable("usage: /budget <n>|unlimited  (e.g. /budget 200000)", is_error=True)
                )
            else:
                self.harness.set_session_budget(self.session, parsed)
                prefs.save(token_budget=parsed)
                self._write(
                    ui.info_line(f"budget set to {parsed or 'unlimited'} for this session and saved as the default")
                )
                self._refresh_status()

    def _theme(self, args: list[str]) -> None:
        if not args or ui.is_query(args):
            names = ", ".join(sorted(self.available_themes))
            self._write(ui.info_line(f"theme: {self.theme}\navailable: {names}"))
            return
        name = args[0]
        if name not in self.available_themes:
            self._write(ui.tool_result_renderable(f"unknown theme {name!r} — /theme with no args lists them", is_error=True))
            return
        self.theme = name  # triggers watch_theme -> persisted automatically
        self._write(ui.info_line(f"theme set to '{name}'"))

    def _copy(self) -> None:
        if not self._last_answer:
            self._write(ui.info_line("no answer to copy yet"))
            return
        self.copy_to_clipboard(self._last_answer)  # Textual OSC-52 clipboard
        self._write(ui.info_line(f"copied {len(self._last_answer)} chars to the clipboard"))

    def _save(self, path: str) -> None:
        from pathlib import Path

        dest = Path(path) if path else Path(f"harness-{str(self.session.id)[:8]}.md")
        dest.write_text(ui.transcript_markdown(self.harness.repo.active_turns(self.session.id)))
        self._write(ui.info_line(f"saved transcript to {dest}"))

    def _mcp(self, args: list[str]) -> None:
        if not args:
            self._write(ui.mcp_renderable(self.harness.tools.mcp_clients))
            return
        direct = "--direct" in args
        rest = [a for a in args if a != "--direct"]
        expose = "direct" if direct else "index"
        try:
            if rest[0] == "http":
                url, name = rest[1], (rest[2] if len(rest) > 2 else "")
                client = self.harness.add_mcp_http(url, name, expose=expose)
                mcpstore.save(client.name, url, expose)  # persist for next launch
                self._write(ui.mcp_connected_renderable(client.name, url, len(client.list_tools())))
            elif rest[0] == "stdio":
                name, command = rest[1], rest[2:]
                client = self.harness.add_mcp_stdio(command, name, expose=expose)
                self._write(ui.mcp_connected_renderable(client.name, " ".join(command), len(client.list_tools())))
            elif rest[0] == "remove" and len(rest) > 1:
                ok = mcpstore.remove(rest[1])
                self._write(ui.info_line(f"removed saved server {rest[1]}") if ok else ui.tool_result_renderable(f"no saved server {rest[1]!r}", is_error=True))
            else:
                self._write(ui.tool_result_renderable("usage: /mcp [http <url> [name]] | [stdio <name> <cmd...>] | [remove <name>] [--direct]", is_error=True))
        except Exception as e:
            self._write(ui.tool_result_renderable(str(e), is_error=True))

    def action_clear_log(self) -> None:
        self.query_one("#log", VerticalScroll).remove_children()


def _version() -> str:
    from .. import __version__

    return __version__


def _backend(harness: Any) -> str:
    from ..persistence import InMemoryRepository

    return "in-memory" if isinstance(harness.repo, InMemoryRepository) else "Postgres"


def run_tui(
    harness: Any, session: Any, external_id: str, provider_label: str, model: str, mcp_lines: list[tuple]
) -> int:
    HarnessApp(harness, session, external_id, provider_label, model, mcp_lines).run()
    return 0
