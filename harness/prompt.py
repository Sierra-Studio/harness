"""System-prompt assembly (the "Soul").

Inspired by Hermes' SOUL.md + layered prompt. The prompt is built from ordered
layers so the stable identity sits first (cache-friendly) and per-call guidance
follows:

    [ soul / identity ]   <- SOUL.md if present & non-empty, else DEFAULT_IDENTITY
    [ tool guidance   ]   <- how to use the built-in tools (incl. when to use Bash)
    [ extra message   ]   <- optional caller-supplied system_message

SOUL.md is loaded fresh; delete it (or its content) to fall back to the default
persona. Comment-only files (HTML comments) count as empty.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_IDENTITY = (
    "You are a capable AI assistant running inside a harness. You are helpful, "
    "direct, and honest. You assist with answering questions, writing and editing "
    "code, analyzing information, and executing actions through your tools. You "
    "communicate clearly, admit uncertainty when appropriate, and prioritize being "
    "genuinely useful over being verbose. Be targeted and efficient."
)

# How the agent should reason about its built-in tools — and, crucially, when to
# reach for Bash. This is the behavioural contract the user asked for: Bash is the
# powerful fallback whenever no specific tool fits but the OS can do the job.
TOOL_GUIDANCE = (
    "# Your tools\n"
    "You have four always-available built-in tools, plus external tools you can "
    "discover on demand:\n"
    "- SearchTools(query): keyword-search the catalog of external tools (from MCP "
    "servers). Use it to find a tool by describing what you need.\n"
    "- GetTools(name): fetch the full input schema of one external tool before "
    "calling it.\n"
    "- GetSkills(query): recall the user's saved skills (reusable procedures).\n"
    "- Bash(command): run a shell command in your per-session sandbox.\n"
    "\n"
    "# Choosing a tool — Bash is your universal fallback\n"
    "Follow this order on every action:\n"
    "1. If one of your skills already covers the task, follow it (GetSkills).\n"
    "2. Otherwise, if a specialized external tool likely exists, find it with "
    "SearchTools and call it — a purpose-built tool beats a raw shell command.\n"
    "3. Otherwise, if the task CAN be done with ordinary operating-system commands "
    "(files, text processing, git, http via curl, package managers, running code, "
    "data wrangling, system inspection), use Bash. Do not refuse or describe what "
    "you would do — actually run the command.\n"
    "Bash is powerful and general: prefer a real tool when one fits, but when none "
    "does and the OS can solve it, Bash is the right answer rather than giving up.\n"
    "\n"
    "# Using Bash well\n"
    "- Your working directory persists across Bash calls within a session; `cd` "
    "into a project once and later commands stay there.\n"
    "- Exported environment variables do NOT persist between calls (each call is a "
    "fresh process). Prefix them inline: `VAR=value some_command`.\n"
    "- Chain related steps with && to keep them atomic; check the reported exit "
    "code and stderr, and fix-and-retry on failure instead of fabricating output.\n"
    "- Keep output focused (use head/tail/grep/wc) — very large output is "
    "truncated.\n"
    "When a tool or command fails and blocks the real path, say so honestly and try "
    "an alternative. Never substitute plausible-looking fabricated results for "
    "output you could not actually produce."
)

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _meaningful(text: str) -> str:
    """Strip HTML comments and whitespace; return '' if nothing remains."""
    stripped = _HTML_COMMENT.sub("", text or "").strip()
    return stripped


def load_soul(soul_path: str = "", *, explicit: str = "") -> str:
    """Resolve the soul/identity text.

    Precedence: explicit string > SOUL.md at soul_path > SOUL.md in CWD >
    DEFAULT_IDENTITY. A comment-only or empty SOUL.md falls through to default.
    """
    if _meaningful(explicit):
        return _meaningful(explicit)

    candidates = []
    if soul_path:
        candidates.append(Path(soul_path))
    candidates.append(Path.cwd() / "SOUL.md")

    for path in candidates:
        try:
            if path.is_file():
                content = _meaningful(path.read_text(encoding="utf-8"))
                if content:
                    return content
        except OSError:
            continue
    return DEFAULT_IDENTITY


def build_system_prompt(soul_path: str = "", *, soul: str = "",
                        tool_guidance: bool = True,
                        extra: str = "") -> str:
    """Assemble the layered system prompt. `extra` is an optional appended block."""
    layers = [load_soul(soul_path, explicit=soul)]
    if tool_guidance:
        layers.append(TOOL_GUIDANCE)
    if _meaningful(extra):
        layers.append(extra.strip())
    return "\n\n".join(layers)
