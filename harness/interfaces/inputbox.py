"""A Claude-Code-style bordered input box for the inline (scrollback) front-end.

`read_boxed()` reads one line of input while drawing a rounded, single-line
input box pinned at the bottom of the terminal — the visual signature of
Claude Code's prompt. Crucially it does this WITHOUT the alternate screen or
mouse capture: the box lives only while we wait for input, and everything the
turn prints scrolls into the terminal's own scrollback above it. That's what
gives the inline front-end fully native scroll / selection / copy.

The editor runs the terminal in raw mode and redraws the box after each
keystroke. It supports the line-editing keys people reach for — arrows, Home/
End, Backspace/Delete, Ctrl-A/E/U/W/K, and ↑/↓ history with a stashed draft —
and horizontally scrolls the visible window when a line outgrows the box.

On anything that can't support raw mode (not a TTY, no termios — e.g. Windows)
it transparently falls back to `console.input()`, so callers never special-case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

try:
    import termios
    import tty

    _RAW_OK = True
except ImportError:  # pragma: no cover - Windows / exotic platforms
    _RAW_OK = False

# ANSI helpers — emitted directly (not via Rich) so in-place cursor math is exact.
_CSI = "\x1b["
_CYAN = "\x1b[36m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"

TL, TR, BL, BR, H, V = "╭", "╮", "╰", "╯", "─", "│"


@dataclass
class _Editor:
    """Mutable state of one line edit: the buffer, the cursor, and — for ↑/↓
    history — the current index into `history` plus the half-typed `draft`
    stashed when the user first arrows up (restored on arrowing back down)."""

    buf: list[str] = field(default_factory=list)
    cur: int = 0
    hist_idx: int | None = None
    draft: str = ""

    @property
    def text(self) -> str:
        return "".join(self.buf)


def _box_width() -> int:
    import shutil

    cols = shutil.get_terminal_size((80, 24)).columns
    # one column short of the terminal so a full-width row never touches the
    # right margin — in raw mode that would autowrap and a following newline
    # would then leave a blank row inside the box.
    return max(20, cols - 1)


def _visible_window(display: str, cur: int, width: int) -> tuple[str, int]:
    """Slice `display` to `width` cells so the cursor stays visible, returning
    the visible substring and the cursor's column within it. Scrolls the window
    with the cursor once the text is longer than the box interior."""
    if len(display) <= width:
        return display, cur
    # keep the cursor roughly a few cells from the right edge as it advances
    start = max(0, min(cur - width + 1, len(display) - width))
    return display[start : start + width], cur - start


def _render(ed: _Editor, prompt: str, out, drawn: bool) -> None:
    """Paint the 3-row box, parking the physical cursor on the middle (text) row.

    The box always occupies exactly three rows with the cursor parked on the
    middle one. So a *redraw* (`drawn=True`) first steps back up to the box's top
    row and clears to end-of-screen, then reprints; the first paint just anchors
    at column 0. Rows are joined with `\\r\\n` (raw mode has no CR translation),
    and `_box_width` keeps them off the right margin, so no row wraps."""
    width = _box_width()
    inner = width - 2  # space between the vertical borders
    field_w = inner - 1 - len(prompt)  # a leading pad space, then prompt, then text
    view, vcur = _visible_window(ed.text, ed.cur, max(1, field_w))
    body = f" {prompt}{view}"
    body = body + " " * (inner - _cellwidth(body))
    top = f"{_CYAN}{TL}{H * inner}{TR}{_RESET}"
    mid = f"{_CYAN}{V}{_RESET}{body}{_CYAN}{V}{_RESET}"
    bot = f"{_CYAN}{BL}{H * inner}{BR}{_RESET}"
    prefix = f"\r{_CSI}1A{_CSI}0J" if drawn else "\r"  # redraw: back to top row + clear
    out.write(prefix + top + "\r\n" + mid + "\r\n" + bot)
    # park cursor on the middle row at the text cursor column
    col = 2 + len(prompt) + vcur  # left border + pad space + prompt + cursor offset
    out.write(f"\r{_CSI}1A{_CSI}{col}C")
    out.flush()


def _cellwidth(s: str) -> int:
    # single-line ASCII/BMP prompt content; treat each char as one cell. (Wide
    # CJK/emoji width is out of scope for v1 — the box may be a cell off for them.)
    return len(s)


def _clear_box(out) -> None:
    """Erase the box before committing a line or running a turn: from the middle
    row, go to the top row and clear everything below."""
    out.write(f"\r{_CSI}1A{_CSI}0J")
    out.flush()


def _read_key(rd) -> str:
    """Read one logical key: a printable char, or a token like 'LEFT', 'UP',
    'HOME', 'DEL', 'ENTER', 'BS', 'C-a', … Parses the common CSI/SS3 escape
    sequences; unknown escapes are swallowed so they never corrupt the buffer."""
    ch = rd(1)
    if not ch:
        return "EOF"
    o = ord(ch)
    if o == 13 or o == 10:
        return "ENTER"
    if o == 127 or o == 8:
        return "BS"
    if o == 3:
        return "C-c"
    if o == 4:
        return "C-d"
    if o == 1:
        return "HOME"
    if o == 5:
        return "END"
    if o == 21:
        return "C-u"
    if o == 11:
        return "C-k"
    if o == 23:
        return "C-w"
    if o == 27:  # escape sequence
        a = rd(1)
        if a in ("[", "O"):
            b = rd(1)
            if b == "A":
                return "UP"
            if b == "B":
                return "DOWN"
            if b == "C":
                return "RIGHT"
            if b == "D":
                return "LEFT"
            if b in ("H",):
                return "HOME"
            if b in ("F",):
                return "END"
            if b == "3":
                rd(1)  # trailing '~'
                return "DEL"
            return "IGN"
        return "IGN"
    if o < 32:
        return "IGN"
    return ch


def read_boxed(console, *, prompt: str = "› ", history: list[str] | None = None) -> str | None:
    """Read one line in a bordered box. Returns the submitted text, or None on
    EOF (Ctrl-D on an empty line). Ctrl-C clears a non-empty line, or aborts the
    read with KeyboardInterrupt when the line is already empty."""
    history = history if history is not None else []
    if not (_RAW_OK and sys.stdin.isatty() and sys.stdout.isatty()):
        # non-tty / unsupported: plain prompt, still native everything
        try:
            return console.input(f"[cyan]{prompt}[/]")
        except EOFError:
            return None

    out = sys.stdout
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    ed = _Editor()
    try:
        tty.setraw(fd)

        def rd(n: int) -> str:
            return sys.stdin.read(n)

        _render(ed, prompt, out, drawn=False)
        while True:
            key = _read_key(rd)
            if key == "ENTER":
                _clear_box(out)
                return ed.text
            if key == "EOF" or key == "C-d":
                if ed.buf:
                    continue
                _clear_box(out)
                return None
            if key == "C-c":
                if ed.buf:
                    ed.buf.clear()
                    ed.cur = 0
                else:
                    _clear_box(out)
                    raise KeyboardInterrupt
            elif key == "BS":
                if ed.cur > 0:
                    del ed.buf[ed.cur - 1]
                    ed.cur -= 1
            elif key == "DEL":
                if ed.cur < len(ed.buf):
                    del ed.buf[ed.cur]
            elif key == "LEFT":
                ed.cur = max(0, ed.cur - 1)
            elif key == "RIGHT":
                ed.cur = min(len(ed.buf), ed.cur + 1)
            elif key == "HOME":
                ed.cur = 0
            elif key == "END":
                ed.cur = len(ed.buf)
            elif key == "C-u":
                del ed.buf[: ed.cur]
                ed.cur = 0
            elif key == "C-k":
                del ed.buf[ed.cur :]
            elif key == "C-w":
                i = ed.cur
                while i > 0 and ed.buf[i - 1] == " ":
                    i -= 1
                while i > 0 and ed.buf[i - 1] != " ":
                    i -= 1
                del ed.buf[i : ed.cur]
                ed.cur = i
            elif key in ("UP", "DOWN"):
                _history_step(ed, history, -1 if key == "UP" else 1)
            elif key == "IGN":
                pass
            else:  # printable
                ed.buf.insert(ed.cur, key)
                ed.cur += 1
            _render(ed, prompt, out, drawn=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _history_step(ed: _Editor, history: list[str], delta: int) -> None:
    """Walk command history like a shell: ↑ from a fresh line stashes the draft,
    ↓ past the newest entry restores it."""
    if not history:
        return
    if ed.hist_idx is None:
        if delta > 0:
            return
        ed.draft = ed.text
        ed.hist_idx = len(history) - 1
    else:
        ed.hist_idx += delta
        if ed.hist_idx >= len(history):
            ed.hist_idx = None
            _set(ed, ed.draft)
            return
        if ed.hist_idx < 0:
            ed.hist_idx = 0
    _set(ed, history[ed.hist_idx])


def _set(ed: _Editor, text: str) -> None:
    ed.buf = list(text)
    ed.cur = len(ed.buf)
