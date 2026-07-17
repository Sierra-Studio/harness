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

import os
import re
import threading
from pathlib import Path
from typing import Any, cast

from rich.markdown import Markdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.suggester import Suggester
from textual.widgets import Header, Input, Static

from ..core import ALLOW, ALWAYS, DENY
from ..core.permissions import next_mode
from . import mcpstore, prefs, ui

# directories skipped in @-path autocomplete / directory expansion
_MENTION_IGNORE = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
_MENTION_MAX_BYTES = 100_000  # per-file cap when inlining an @mention's contents
_MENTION_MAX_FILES = 25  # cap when an @mention points at a directory


def _mention_partial(value: str) -> str | None:
    """The active @-mention being typed (text after the last space that starts
    with `@`), or None if the cursor isn't in an @token. Returns the part after
    the `@`, so `look @app/ma` -> `app/ma` and `@` -> `` (list everything)."""
    token = value.rpartition(" ")[2]
    return token[1:] if token.startswith("@") else None


def _list_path_matches(root: Path, partial: str, limit: int = 10) -> list[str]:
    """Filesystem completions for a partial @-path, relative to `root`. Each is a
    completed relative path with a trailing `/` for directories. Directories sort
    first, then files, alphabetically; ignore-listed names are skipped."""
    if partial.endswith("/"):
        directory, name = partial, ""
    else:
        p = Path(partial)
        directory = "" if p.parent == Path(".") else f"{p.parent}/"
        name = p.name
    base = root / directory if directory else root
    try:
        entries = sorted(base.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError:
        return []
    low = name.lower()
    out: list[str] = []
    for e in entries:
        if e.name in _MENTION_IGNORE:
            continue
        if e.name.lower().startswith(low):
            out.append(f"{directory}{e.name}" + ("/" if e.is_dir() else ""))
            if len(out) >= limit:
                break
    return out


def _collect_dir_files(directory: Path, root: Path, limit: int = _MENTION_MAX_FILES) -> list[str]:
    """Relative paths of files under `directory` (recursive), pruning ignore-listed
    subdirs and stopping at `limit` files. Used to expand an @dir/ mention."""
    out: list[str] = []
    for cur, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in _MENTION_IGNORE)
        for fn in sorted(files):
            if fn in _MENTION_IGNORE:
                continue
            out.append(os.path.relpath(os.path.join(cur, fn), root))
            if len(out) >= limit:
                return out
    return out


class InputSuggester(Suggester):
    """Ghost-text completion for `/commands`. `@file` mentions get a richer
    multi-match dropdown instead (see HarnessApp._ac_*), so this stays out of
    their way and only completes slash commands."""

    def __init__(self, commands: list[str]) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._commands = commands

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        low = value.lower()
        return next((c for c in self._commands if c.lower().startswith(low)), None)


def _os_clipboard_write(text: str) -> str | None:
    """Write text to the OS clipboard via a local tool, more reliable than OSC-52.

    OSC-52 (Textual's `copy_to_clipboard`) is silently dropped by some terminals
    (e.g. the VS Code integrated terminal), so we also pipe to the platform's
    clipboard command when one is present. Returns the tool name used, or None
    if none is available (in which case OSC-52 remains the only path)."""
    import shutil
    import subprocess
    import sys

    if sys.platform == "darwin":
        candidates = [["pbcopy"]]
    elif sys.platform == "win32":
        candidates = [["clip"]]
    else:
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text.encode(), check=True)
            return cmd[0]
        except (OSError, subprocess.CalledProcessError):
            continue
    return None

# slash commands offered as ghost-text autocomplete (ordered by how completions resolve)
_SLASH_NAMES = [
    "/help", "/session", "/sessions", "/skills", "/skills add", "/tools", "/mcp",
    "/resume", "/retry", "/copy", "/save", "/persona", "/system-prompt", "/model",
    "/budget", "/theme", "/mode", "/auto", "/plan", "/manual", "/clear", "/new", "/exit", "/quit",
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
        """Tab accepts the highlighted @file match if the dropdown is open,
        else the ghost-text (slash-command) suggestion."""
        app = cast("HarnessApp", self.app)
        if app._ac_open and app._ac_accept():
            return
        suggestion = self._suggestion
        if suggestion and len(suggestion) > len(self.value):
            self.value = suggestion
            self.cursor_position = len(self.value)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The suggester (slash commands + @file paths) is installed at mount,
        # once the app knows the workspace root — see HarnessApp.on_mount.
        super().__init__(*args, **kwargs)
        self.history: list[str] = []
        self._idx: int | None = None
        self._draft = ""

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._idx = None
        self._draft = ""

    def action_history_prev(self) -> None:
        # While the @file dropdown is open, ↑/↓ move its selection instead.
        app = cast("HarnessApp", self.app)
        if app._ac_open:
            app._ac_move(-1)
            return
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
        app = cast("HarnessApp", self.app)
        if app._ac_open:
            app._ac_move(1)
            return
        if self._idx is None:
            return
        if self._idx < len(self.history) - 1:
            self._idx += 1
            self.value = self.history[self._idx]
        else:
            self._idx = None
            self.value = self._draft
        self.cursor_position = len(self.value)


class PermissionBar(Static):
    """Inline manual-mode approval prompt shown below the input (not a modal).

    Hidden until `ask()` reveals it and grabs focus; the y/a/n/esc keys resolve
    the pending decision through a resolver callback, then focus returns to the
    input. One prompt is live at a time (the loop blocks on each call)."""

    can_focus = True

    BINDINGS = [
        Binding("y", "decide('allow')", show=False),
        Binding("a", "decide('always')", show=False),
        Binding("n", "decide('deny')", show=False),
        Binding("escape", "decide('deny')", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._resolver: Any = None

    def ask(self, display: str, summary: str, resolver: Any, *, hint: str | None = None) -> None:
        self._resolver = resolver
        body = Text()
        body.append("permission  ", style="bold")
        body.append(display, style="bold")
        if summary:
            body.append(f"  {summary}", style="dim")
        body.append("\n")
        body.append(hint or "[y] allow once   [a] allow all session   [n]/esc deny", style="dim")
        self.update(body)
        self.display = True
        self.focus()

    def action_decide(self, decision: str) -> None:
        self.display = False
        resolver, self._resolver = self._resolver, None
        if resolver is not None:
            resolver(decision)


class SelectableStatic(Static):
    """A `Static` that stays mouse-selectable even when it renders a rich
    renderable (Markdown, panels, tables).

    Textual can only extract text from a widget when it renders to `Text`/
    `Content`; a `Static(Markdown(...))` returns `None` from `get_selection`, so
    dragging over an agent answer highlights nothing and copies nothing. We fix
    that by carrying the message's source text and handing it to Textual on
    demand. Textual can't map sub-offsets inside a complex renderable, so such a
    drag is a whole-widget selection — for a chat message that simply means the
    whole answer is selected and copied, which is what people want anyway. Plain
    Text/str content keeps normal per-character selection via `super()`."""

    def __init__(self, renderable: Any, *, select_text: str = "", **kwargs: Any) -> None:
        super().__init__(renderable, **kwargs)
        self._select_text = select_text

    def set_content(self, renderable: Any, select_text: str) -> None:
        """Update both the rendered content and the text copied on selection —
        used while an answer streams in chunk by chunk."""
        self._select_text = select_text
        self.update(renderable)

    def get_selection(self, selection: Any) -> tuple[str, str] | None:
        if self._select_text:
            return selection.extract(self._select_text), "\n"
        return super().get_selection(selection)


class HarnessApp(App):
    CSS = """
    #log { height: 1fr; padding: 0 1; }
    #log > * { width: 1fr; height: auto; }
    #log > .turn-start { margin-top: 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
    #bottom { dock: bottom; height: auto; margin-bottom: 1; }
    #prompt { height: 5; border: none;
              border-top: tall $warning; border-bottom: tall $warning; padding: 1 1; }
    #prompt:focus { border: none; border-top: tall $warning; border-bottom: tall $warning; }
    #perm { height: auto; padding: 0 1; margin-top: 1; display: none;
            border: round $warning; }
    #perm:focus { border: round $warning; }
    #ac { height: auto; max-height: 10; padding: 0 1; display: none;
          background: $panel; border: round $primary; }
    #mode-hint { height: 1; padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        ("ctrl+c", "interrupt", "stop / quit"),
        ("escape", "interrupt", "stop"),
        ("ctrl+d", "quit", "quit"),
        ("ctrl+l", "clear_log", "clear"),
        Binding("shift+tab", "toggle_mode", "cycle mode", priority=True),
        # Keyboard scrolling of the history — mouse capture is off (so drag-select
        # + copy work), which also disables wheel scroll, so these stand in for it.
        Binding("pageup", "scroll_log('page_up')", "scroll up", show=False),
        Binding("pagedown", "scroll_log('page_down')", "scroll down", show=False),
        Binding("ctrl+up", "scroll_log('up')", show=False),
        Binding("ctrl+down", "scroll_log('down')", show=False),
        Binding("ctrl+home", "scroll_log('home')", show=False),
        Binding("ctrl+end", "scroll_log('end')", show=False),
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
        self._md_widget: SelectableStatic | None = None
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
        self._root = Path(os.getcwd())  # workspace root for @mentions (refined at mount)
        self._ac_open = False  # @file dropdown visible?
        self._ac_matches: list[str] = []  # current dropdown entries (completed rel paths)
        self._ac_sel = 0  # highlighted index in the dropdown
        # An in-flight AskUser prompt: (threading.Event, box) while the turn's
        # worker thread is blocked waiting for the human's typed answer; None
        # otherwise. Set/cleared by _ask_user (worker) and _submit (UI thread).
        self._asking: tuple | None = None
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
        with Vertical(id="bottom"):
            yield Static("", id="ac")  # @file match dropdown, above the input
            yield PromptInput(placeholder="Message…   (/help · @file · ↑ history · ctrl+c stop)", id="prompt")
            yield PermissionBar(id="perm")
            yield Static("", id="mode-hint")

    def on_mount(self) -> None:
        self.title = "harness"
        # Manual mode asks before each side-effecting tool call; the asker blocks
        # the turn's worker thread while an inline bar below the input collects
        # the y/a/n answer on the UI thread.
        self.harness.permissions.asker = self._ask_permission
        # AskUser pauses the turn mid-flight for free-form human input; the
        # prompter blocks the worker thread while the answer is typed into the
        # (re-enabled) main input on the UI thread. See _ask_user / _submit.
        self.harness.tools.prompter = self._ask_user
        # @file mentions and /commands complete against the workspace root (the
        # folder chat was launched in, same as Bash/Write/Edit resolve against).
        self._root = Path(getattr(self.harness.sandbox, "workspace", None) or os.getcwd())
        self.query_one("#prompt", PromptInput).suggester = InputSuggester(_SLASH_NAMES)
        self.sub_title = self.external_id
        self._refresh_mode_hint()
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

    def get_system_commands(self, screen):
        """Drop Textual's built-in "Maximize/Minimize the widget" palette entries
        — this is a single-pane chat, so widget zooming has no meaning here."""
        for command in super().get_system_commands(screen):
            if command.title in ("Maximize", "Minimize"):
                continue
            yield command

    def copy_to_clipboard(self, text: str) -> None:
        """Route Textual's built-in selection copy to the real OS clipboard.
        Textual's own implementation only emits OSC-52, which several terminals
        drop (macOS Terminal, the VS Code terminal), so we also pipe to
        pbcopy/wl-copy/xclip — the same reliable path /copy uses. super() still
        runs so OSC-52-capable terminals keep working too."""
        super().copy_to_clipboard(text)
        _os_clipboard_write(text)

    def on_text_selected(self, event: Any) -> None:
        """Auto-copy the moment a mouse drag-selection ends, so copying works
        with the mouse *alone*. We can't bind a copy key — ctrl+c is taken for
        interrupt and super+c only exists on macOS — and Textual doesn't copy on
        its own, so without this a drag would highlight but never reach the
        clipboard. Mirrors the X11/primary-selection "select = copy" habit; a
        plain click posts this too but selects nothing, so it's a no-op then."""
        text = self.screen.get_selected_text()
        if text:
            self.copy_to_clipboard(text)
            # transient toast, not a log line — a selection can happen constantly
            self.notify(f"copied {len(text)} chars", timeout=2)

    # -- rendering helpers --------------------------------------------------
    def _write(self, renderable: Any, *, turn_start: bool = False, select_text: str = "") -> SelectableStatic:
        log = self.query_one("#log", VerticalScroll)
        widget = SelectableStatic(renderable, select_text=select_text)
        # A single blank row separates conversational turns; a turn's own
        # outputs (tool calls, results, the reply) sit tight beneath it.
        if turn_start:
            widget.add_class("turn-start")
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

    # -- @file autocomplete dropdown ---------------------------------------
    @on(Input.Changed, "#prompt")
    def _on_input_changed(self, event: Input.Changed) -> None:
        self._ac_update(event.value)

    def _ac_update(self, value: str) -> None:
        """Recompute the @file dropdown from the current input value."""
        partial = _mention_partial(value)
        matches = _list_path_matches(self._root, partial) if partial is not None else []
        if not matches:
            self._ac_close()
            return
        self._ac_matches = matches
        self._ac_sel = min(self._ac_sel, len(matches) - 1) if self._ac_open else 0
        self._ac_open = True
        self._render_ac()

    def _render_ac(self) -> None:
        text = Text()
        for i, m in enumerate(self._ac_matches):
            text.append(f"@{m}\n", style="reverse" if i == self._ac_sel else "")
        self.query_one("#ac", Static).update(text)
        self.query_one("#ac", Static).display = True

    def _ac_close(self) -> None:
        if self._ac_open or self.query_one("#ac", Static).display:
            self._ac_open = False
            self._ac_matches = []
            self._ac_sel = 0
            self.query_one("#ac", Static).display = False

    def _ac_move(self, delta: int) -> None:
        if self._ac_open and self._ac_matches:
            self._ac_sel = (self._ac_sel + delta) % len(self._ac_matches)
            self._render_ac()

    def _ac_accept(self) -> bool:
        """Replace the active @token with the highlighted match. Returns True if
        it consumed the key. A directory keeps the dropdown open (drill in); a
        file inserts it plus a trailing space and closes the dropdown."""
        if not self._ac_open or not self._ac_matches:
            return False
        inp = self.query_one("#prompt", PromptInput)
        head = inp.value.rpartition(" ")[0]
        prefix = f"{head} " if head else ""
        chosen = self._ac_matches[self._ac_sel]
        if chosen.endswith("/"):
            inp.value = f"{prefix}@{chosen}"
            inp.cursor_position = len(inp.value)
            self._ac_update(inp.value)  # list the directory's contents
        else:
            inp.value = f"{prefix}@{chosen} "
            inp.cursor_position = len(inp.value)
            self._ac_close()
        return True

    # -- input --------------------------------------------------------------
    @on(Input.Submitted, "#prompt")
    def _submit(self, event: Input.Submitted) -> None:
        # An AskUser prompt is in flight: this Enter is the human's answer, not a
        # new message. (The input is only enabled mid-turn because _ask_user
        # re-enabled it, so a submit here is unambiguously the answer.) Hand it
        # back to the blocked worker thread and stop.
        if self._asking is not None:
            self._answer_ask(event.value.strip())
            event.input.value = ""
            return
        # Enter picks the highlighted @file match when the dropdown is open,
        # rather than sending the message (the input keeps the completed value).
        if self._ac_open and self._ac_accept():
            return
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
        self._write(ui.user_line(msg), turn_start=True)
        if msg.startswith("/"):
            self._slash(msg)
            return
        self._begin_turn(msg)

    def _begin_turn(self, msg: str) -> None:
        self._busy = True
        self._cancel = False
        self._last_user = msg  # raw text (with @mentions) — re-expanded on /retry
        self._answer = []
        # Inline any @file mentions so the model sees their contents; the log
        # keeps showing your original text (already written by the caller).
        sent, attached = self._expand_mentions(msg)
        if attached:
            self._write(ui.info_line("attached " + ", ".join(attached)))
        self._label = "thinking…"
        self.query_one("#prompt", Input).disabled = True
        self._refresh_status()
        self._run_turn(sent)

    def _expand_mentions(self, msg: str) -> tuple[str, list[str]]:
        """Inline @path mentions as context appended to the message. A file adds
        its contents; a directory adds every file under it (recursively, pruning
        ignore-listed dirs, capped at `_MENTION_MAX_FILES`). Returns (message_to_
        send, attached_paths); non-file/unreadable mentions stay as plain text."""
        root = getattr(self, "_root", None) or Path(os.getcwd())
        blocks: list[str] = []
        attached: list[str] = []

        def add_file(rel: str) -> None:
            if rel in attached:
                return
            try:
                text = (root / rel).read_text(errors="replace")
            except OSError:
                return
            if len(text) > _MENTION_MAX_BYTES:
                text = text[:_MENTION_MAX_BYTES] + "\n… [truncated]"
            blocks.append(f'<file path="{rel}">\n{text}\n</file>')
            attached.append(rel)

        for raw in re.findall(r"@(\S+)", msg):
            rel = raw.rstrip(".,;:!?)]}").rstrip("/")  # trim trailing punctuation / slash
            path = root / rel
            if path.is_dir():
                for f in _collect_dir_files(path, root):
                    add_file(f)
            elif path.is_file():
                add_file(rel)
        if not blocks:
            return msg, []
        return msg + "\n\n" + "\n".join(blocks), attached

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
            md = "".join(self._md_buf)
            if self._md_widget is None:
                # select_text=md so a drag over the answer selects+copies its
                # markdown source (Markdown renderables aren't selectable otherwise).
                self._md_widget = self._write(Markdown(md), select_text=md)
            else:
                self._md_widget.set_content(Markdown(md), md)
                self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        elif ev.kind == "tool_start":
            self._end_md()
            name = ui.tool_display_name(ev.name, ev.args)
            if ev.name == "RenderUI":
                widget = ui.ui_renderable(ev.args.get("root"))
            elif ev.name == "ExitPlanMode":
                widget = ui.plan_renderable(ev.args.get("plan", ""))
            else:
                widget = None
            if widget is not None:
                self._write(widget)
                if ev.name == "RenderUI":
                    # ExitPlanMode is deliberately NOT added to _rendered_ui —
                    # the normal tool_result line below still prints the
                    # approve/reject outcome, unlike RenderUI's pure ack noise.
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

    # -- permission mode ----------------------------------------------------
    def _refresh_mode_hint(self) -> None:
        """Persistent indicator line under the input (Claude-Code style)."""
        mode = self.harness.permissions.mode
        if mode == "manual":
            text = "[] manual mode · approve each tool  (shift+tab to cycle)"
        elif mode == "plan":
            text = "plan mode · read-only, call ExitPlanMode when ready  (shift+tab to cycle)"
        else:
            text = ">> auto mode · run all tools  (shift+tab to cycle)"
        self.query_one("#mode-hint", Static).update(text)

    def _set_mode(self, mode: str) -> None:
        mode = self.harness.set_permission_mode(mode)
        prefs.save(permission_mode=mode)
        self._refresh_mode_hint()

    def action_toggle_mode(self) -> None:
        """shift+tab: cycle auto -> plan -> manual -> auto, reflected in the
        bottom hint line."""
        self._set_mode(next_mode(self.harness.permissions.mode))

    def _ask_permission(self, name: str, args: dict) -> str:
        """Called from the turn worker thread. Reveal the inline permission bar
        below the input on the UI thread and block until the user answers, then
        return ALLOW / ALWAYS / DENY and hand focus back to the input.

        ExitPlanMode gets its own prompt text (the plan itself is already
        visible above, rendered by _handle's tool_start branch) and, on
        approval, flips the mode to manual right here."""
        if name == "ExitPlanMode":
            display = "approve this plan and start implementing?"
            plan = args.get("plan", "")
            summary = f"{len(plan.splitlines())} line plan above" if plan.strip() else ""
            hint = "[y] approve   [n]/esc keep refining"
        else:
            display = ui.tool_display_name(name, args)
            summary = ui.tool_args_str(name, args)
            hint = None
        event = threading.Event()
        box = {"decision": DENY}

        def show() -> None:
            def resolver(decision: str | None) -> None:
                box["decision"] = decision or DENY
                self.query_one("#prompt", Input).focus()
                event.set()

            self.query_one("#perm", PermissionBar).ask(display, summary, resolver, hint=hint)

        self.call_from_thread(show)
        event.wait()
        decision = box["decision"]
        if name == "ExitPlanMode" and decision in (ALLOW, ALWAYS):
            # NOT self.harness.set_permission_mode(...) — that would eagerly
            # deregister ExitPlanMode from the registry before
            # ToolRegistry.dispatch looks it up for THIS call. Flip the mode
            # only; AgentLoop._run_steps resyncs the registry at the top of
            # the next step, once dispatch of this approved call has
            # finished. See sync_plan_mode_tool's docstring.
            def land_in_manual() -> None:
                self.harness.permissions.set_mode("manual")
                prefs.save(permission_mode="manual")
                self._refresh_mode_hint()

            self.call_from_thread(land_in_manual)
        return decision

    def _ask_user(self, question: str, meta: dict) -> str:
        """Called from the turn worker thread (AskUser tool). Render the question
        in the log and re-enable the main input to collect a free-form answer on
        the UI thread, blocking the worker until the human submits it. Returns the
        typed answer (empty string if interrupted)."""
        event = threading.Event()
        box = {"answer": ""}

        def show() -> None:
            self._end_md()  # flush any streamed markdown before the question panel
            self._write(ui.ask_renderable(question, meta.get("options")))
            self._asking = (event, box)
            inp = self.query_one("#prompt", Input)
            inp.disabled = False
            inp.placeholder = "Type your answer…   (enter to send)"
            inp.focus()

        self.call_from_thread(show)
        event.wait()
        return box["answer"]

    def _answer_ask(self, answer: str) -> None:
        """UI thread: deliver the human's AskUser answer back to the blocked
        worker thread and restore the input to its normal (disabled, mid-turn)
        state."""
        pending, self._asking = self._asking, None
        if pending is None:
            return
        event, box = pending
        box["answer"] = answer
        self._write(ui.user_line(answer or "(no answer)"))
        inp = self.query_one("#prompt", Input)
        inp.disabled = True
        inp.placeholder = "Message…   (/help · @file · ↑ history · ctrl+c stop)"
        event.set()

    def action_interrupt(self) -> None:
        """Ctrl+C / Esc: answer a pending AskUser with an empty string first (so
        the blocked worker unblocks), else dismiss the @file dropdown, else stop a
        running turn, else quit the app."""
        if self._asking is not None:
            self._answer_ask("")
            return
        if self._ac_open:
            self._ac_close()
            return
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
                self._write(ui.user_line(self._last_user), turn_start=True)
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
        elif cmd in ("/auto", "/manual", "/plan"):
            self._set_mode(cmd[1:])
        elif cmd == "/mode":
            self._write(ui.info_line(
                f"permission mode: {self.harness.permissions.mode}  (/auto · /plan · /manual · shift+tab)"
            ))
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
            model = args[0]
            available = self.harness.provider.available_models()
            if available is not None and model not in available:
                import difflib

                near = difflib.get_close_matches(model, available, n=3, cutoff=0.4)
                near += [m for m in available if model.lower() in m.lower() and m not in near]
                hint = f"  did you mean: {', '.join(near[:3])}?" if near else ""
                self._write(ui.tool_result_renderable(f"unknown model '{model}' — not saved.{hint}", is_error=True))
                return
            self.harness.set_session_model(self.session, model)
            prefs.save(model=model)
            self._write(ui.info_line(f"model set to '{model}' for this session and saved as the default"))
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
        self.copy_to_clipboard(self._last_answer)  # OSC-52; may be a no-op in some terminals
        via = _os_clipboard_write(self._last_answer)  # local tool (pbcopy/xclip/…) is reliable
        note = "" if via is None else f" (via {via})"
        self._write(ui.info_line(f"copied {len(self._last_answer)} chars to the clipboard{note}"))

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

    def action_scroll_log(self, how: str) -> None:
        """Scroll the message history from the keyboard (PgUp/PgDn, Ctrl+↑/↓,
        Ctrl+Home/End) — stands in for the wheel, which needs mouse capture."""
        log = self.query_one("#log", VerticalScroll)
        {
            "page_up": log.scroll_page_up,
            "page_down": log.scroll_page_down,
            "up": log.scroll_up,
            "down": log.scroll_down,
            "home": log.scroll_home,
            "end": log.scroll_end,
        }[how](animate=False)


def _version() -> str:
    from .. import __version__

    return __version__


def _backend(harness: Any) -> str:
    from ..persistence import InMemoryRepository

    return "in-memory" if isinstance(harness.repo, InMemoryRepository) else "Postgres"


def run_tui(
    harness: Any, session: Any, external_id: str, provider_label: str, model: str, mcp_lines: list[tuple]
) -> int:
    # Mouse tracking ON (the default) so the wheel scrolls the history. Copy
    # still works two ways: Textual's own drag-to-select + Cmd/Ctrl+C (which
    # HarnessApp.copy_to_clipboard routes to the real OS clipboard, not just
    # OSC-52), and — in any terminal — holding Option (macOS) / Shift (most
    # others) while dragging to force the terminal's native selection + copy.
    HarnessApp(harness, session, external_id, provider_label, model, mcp_lines).run()
    return 0
