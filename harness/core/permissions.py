"""Tool-call permission policy — the "manual mode" / "auto mode" gate.

`Permissions` decides whether a tool call may run before the loop dispatches it:

  - "auto"   : everything runs, no prompting (the default — all commands approved).
  - "manual" : each side-effecting tool call is put to an `asker` callback that
               the interface (TUI/CLI) supplies; the human answers allow / deny.

Read-only tools (`READONLY`) are never prompted, so manual mode stays usable.
The gate is interface-agnostic: `asker` blocks the loop until it returns, so the
TUI can pop a modal and the plain CLI can read a line, both synchronously. When
no `asker` is set (e.g. piped, non-interactive) manual mode falls back to allow,
since there is no human to ask and blocking would hang the turn.
"""

from __future__ import annotations

from collections.abc import Callable

# Answers an `asker` may return. ALLOW/DENY are one-shot; ALWAYS grants the tool
# for the rest of the session (remembered per tool name, like Claude Code's "a").
ALLOW = "allow"
ALWAYS = "always"
DENY = "deny"

# Pure read/observe tools: no side effects, so never worth a prompt.
READONLY = frozenset({"SearchTools", "GetTools", "SearchSkills", "GetSkill", "RenderUI"})

# Callback the interface installs: (tool_name, args) -> ALLOW | ALWAYS | DENY.
Asker = Callable[[str, dict], str]


class Permissions:
    def __init__(self, mode: str = "auto", asker: Asker | None = None):
        self.mode = mode if mode in ("auto", "manual") else "auto"
        self.asker = asker
        self._session_allow: set[str] = set()  # tools granted ALWAYS this session

    def set_mode(self, mode: str) -> str:
        """Switch mode; returns the effective mode. Unknown values are ignored."""
        if mode in ("auto", "manual"):
            self.mode = mode
        return self.mode

    def toggle(self) -> str:
        """Flip auto<->manual; returns the new mode (for a shift+tab-style key)."""
        return self.set_mode("manual" if self.mode == "auto" else "auto")

    def check(self, name: str, args: dict) -> bool:
        """True if the call may run. In manual mode, prompts via `asker` for any
        tool that isn't read-only or already session-approved."""
        if self.mode == "auto" or name in READONLY or name in self._session_allow:
            return True
        if self.asker is None:  # no human to ask (non-interactive) — don't hang
            return True
        decision = self.asker(name, args)
        if decision == ALWAYS:
            self._session_allow.add(name)
            return True
        return decision == ALLOW
