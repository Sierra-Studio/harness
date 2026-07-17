"""Tool-call permission policy — the "auto" / "plan" / "manual" mode gate.

`Permissions` decides whether a tool call may run before the loop dispatches it:

  - "auto"   : everything runs, no prompting (the default — all commands approved).
  - "plan"   : read-only research mode. Only READONLY tools, heuristically
               read-only `Bash` commands, and `ExitPlanMode` may run; everything
               else is silently blocked (no prompt — the model should just try
               something else). `ExitPlanMode` itself always goes through the
               `asker`, since presenting a plan for human approval is the whole
               point of the mode.
  - "manual" : each side-effecting tool call is put to an `asker` callback that
               the interface (TUI/CLI) supplies; the human answers allow / deny.

Read-only tools (`READONLY`) are never prompted, so manual mode stays usable.
The gate is interface-agnostic: `asker` blocks the loop until it returns, so the
TUI can pop a modal and the plain CLI can read a line, both synchronously. When
no `asker` is set (e.g. piped, non-interactive) manual mode falls back to allow,
since there is no human to ask and blocking would hang the turn.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Answers an `asker` may return. ALLOW/DENY are one-shot; ALWAYS grants the tool
# for the rest of the session (remembered per tool name, like Claude Code's "a").
ALLOW = "allow"
ALWAYS = "always"
DENY = "deny"

# Pure read/observe tools: no side effects, so never worth a prompt. AskUser is
# here too — it IS the human interaction (it prompts the user itself), so gating
# it at the permission layer would double-prompt.
READONLY = frozenset(
    {"SearchTools", "GetTools", "SearchSkills", "GetSkill", "RenderUI", "AskUser"}
)

# The mode cycle a shift+tab-style key walks through.
MODE_ORDER = ("auto", "plan", "manual")


def next_mode(mode: str) -> str:
    """The next mode in the auto -> plan -> manual -> auto cycle."""
    i = MODE_ORDER.index(mode) if mode in MODE_ORDER else -1
    return MODE_ORDER[(i + 1) % len(MODE_ORDER)]


# ---- plan-mode Bash classifier -------------------------------------------
#
# This harness has no dedicated read-only file tool (no Read/Grep/Glob) — Bash
# is the *only* way to read a file, list a directory, or grep. Blocking Bash
# outright in plan mode would make plan mode unable to research anything, so
# instead a small allowlist of read-only-looking commands is let through.
#
# This is a best-effort UX guardrail, NOT a security boundary: there is no
# OS-level read-only sandbox here (see LocalSubprocessSandbox in sandbox.py, a
# plain subprocess exec), and an adversarial model could still bypass this
# classifier (e.g. `python -c "open('f','w').write('x')"` — `python` isn't
# allowlisted, so that particular example is actually still blocked, but the
# general point holds: this cuts prompt friction for obviously-safe exploration,
# it does not enforce safety). Matches the spirit, not the rigor, of Claude
# Code's own plan-mode Bash handling.
_SAFE_BASH_BINARIES = frozenset({
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "find", "wc", "pwd",
    "echo", "printf", "which", "file", "stat", "tree", "diff", "du", "whoami", "date",
    "sort", "uniq", "cut", "jq",
})
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "blame", "branch", "remote", "ls-files",
    "rev-parse", "describe",
})
_CONTROL_OPERATORS = re.compile(r"&&|\|\||;|\||\n")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRITE_REDIRECT = re.compile(r">{1,2}(?!&)")  # `>`/`>>`, but not `2>&1`-style fd dup
_SUBSHELL_OR_SUBST = re.compile(r"\$\(|`|<\(")  # command substitution / process substitution


def is_readonly_bash(command: str) -> bool:
    """True only for commands that look confidently read-only.

    Every segment (split on ;, &&, ||, |, newline) must lead with a binary from
    a small read-only allowlist (bare `git` further restricted to a few safe
    subcommands), and the command must contain no write-redirection and no
    command/process substitution (`$(...)`, backticks, `<(...)`) — those can
    hide an arbitrary mutating command inside an otherwise-safe-looking line,
    and this classifier makes no attempt to parse into them. Default-deny:
    anything not confidently recognized is rejected. See the module-level note
    above for why this is a heuristic, not a sandbox.
    """
    command = (command or "").strip()
    if not command or _WRITE_REDIRECT.search(command) or _SUBSHELL_OR_SUBST.search(command):
        return False
    segments = _CONTROL_OPERATORS.split(command)
    if not segments:
        return False
    for seg in segments:
        tokens = seg.strip().split()
        while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
            tokens.pop(0)
        if not tokens:
            return False
        binary = tokens[0].rsplit("/", 1)[-1]
        if binary == "git":
            if len(tokens) < 2 or tokens[1] not in _SAFE_GIT_SUBCOMMANDS:
                return False
            continue
        if binary not in _SAFE_BASH_BINARIES:
            return False
    return True


# Callback the interface installs: (tool_name, args) -> ALLOW | ALWAYS | DENY.
Asker = Callable[[str, dict], str]


class Permissions:
    def __init__(self, mode: str = "auto", asker: Asker | None = None):
        self.mode = mode if mode in MODE_ORDER else "auto"
        self.asker = asker
        self._session_allow: set[str] = set()  # tools granted ALWAYS this session

    def set_mode(self, mode: str) -> str:
        """Switch mode; returns the effective mode. Unknown values are ignored."""
        if mode in MODE_ORDER:
            self.mode = mode
        return self.mode

    def toggle(self) -> str:
        """Cycle auto -> plan -> manual -> auto; returns the new mode (for a
        shift+tab-style key)."""
        return self.set_mode(next_mode(self.mode))

    def check(self, name: str, args: dict) -> bool:
        """True if the call may run.

        In manual mode, prompts via `asker` for any tool that isn't read-only
        or already session-approved. In plan mode, only READONLY tools,
        heuristically read-only Bash, and ExitPlanMode may run — everything
        else is blocked without a prompt; ExitPlanMode itself is put to the
        asker like a manual-mode call, since a human decision is the point.
        """
        if self.mode == "auto" or name in READONLY or name in self._session_allow:
            return True
        if self.mode == "plan" and name != "ExitPlanMode":
            # blanket block, no prompt — model should try something else
            return name == "Bash" and is_readonly_bash(str(args.get("command", "")))
        if self.asker is None:  # no human to ask (non-interactive) — don't hang
            return True
        decision = self.asker(name, args)
        if decision == ALWAYS:
            # ExitPlanMode is excluded from session-wide ALWAYS: it would mean
            # the first plan approval silently pre-approves every later,
            # unrelated plan this session. ALWAYS on it behaves like a one-off
            # ALLOW instead — the next ExitPlanMode call still prompts.
            if name != "ExitPlanMode":
                self._session_allow.add(name)
            return True
        return decision == ALLOW
