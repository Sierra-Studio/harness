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
from datetime import date as _date
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
    "You have six always-available built-in tools, plus external tools you can "
    "discover on demand:\n"
    "- SearchTools(query): keyword-search the catalog of external tools (from MCP "
    "servers). Use it to find a tool by describing what you need.\n"
    "- GetTools(name): fetch the full input schema of one external tool before "
    "calling it.\n"
    "- CallTool(name, arguments): actually RUN an external tool. External tools "
    "are NOT callable directly by name — CallTool is the only way to invoke one. "
    "The flow is always SearchTools -> GetTools -> CallTool.\n"
    "- GetSkill(name): load the full steps of one saved skill (from the \"Your "
    "saved skills\" list below, when present) before following it.\n"
    "- SearchSkills(query): keyword-search saved skills — only needed when the "
    "list below is truncated or you want to search a large set; returns name + "
    "summary for each.\n"
    "- Bash(command): run a shell command in your per-session sandbox.\n"
    "\n"
    "# Choosing a tool — Bash is your universal fallback\n"
    "Follow this order on every action:\n"
    "1. If one of your saved skills (listed under \"Your saved skills\", when "
    "present) covers the task, load its steps with GetSkill(name) and follow "
    "them.\n"
    "2. Otherwise, if a specialized external tool likely exists, find it with "
    "SearchTools and call it — a purpose-built tool beats a raw shell command. "
    "Query in ENGLISH with broad capability terms and synonyms (the tool catalog "
    "is indexed in English), and if the first query returns nothing, REFORMULATE "
    "with related terms before giving up — e.g. for 'gravações/recordings de "
    "reunião' try 'meeting', 'meeting transcript', 'calls', 'recordings'. Data "
    "that clearly lives in an external service (meetings, calendar, email, chat, "
    "tickets, docs) almost always has a tool — do NOT fall back to Bash for it "
    "just because one keyword missed; Bash cannot see a SaaS account.\n"
    "3. Otherwise, if the task CAN be done with ordinary operating-system commands "
    "(files, text processing, git, http via curl, package managers, running code, "
    "data wrangling, system inspection), use Bash. Do not refuse or describe what "
    "you would do — actually run the command.\n"
    "Bash is powerful and general: prefer a real tool when one fits, but when none "
    "does and the OS can solve it, Bash is the right answer rather than giving up.\n"
    "\n"
    "# Closing the loop — once you've found a tool, CALL it\n"
    "- After GetTools returns a schema, your next action is CallTool(name, "
    "arguments) for that tool. Do NOT re-run SearchTools/GetTools for a tool you "
    "already found, and never try to invoke a tool by emitting its name directly "
    "or by stuffing arguments into SearchTools — only CallTool runs it.\n"
    "- Parameters whose schema marks them optional (a default, or a type that "
    "allows null) can be OMITTED — pass arguments={} if you have nothing to "
    "supply. Don't stall hunting for values you don't have; call the tool with "
    "what you know and refine only if the result tells you to.\n"
    "- Never shell out to compute the date or time: today's date is already given "
    "to you at the end of this prompt. Use it directly for any date parameter.\n"
    "- If you catch yourself repeating an identical call, stop and change approach "
    "rather than looping.\n"
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
    """Assemble the layered system prompt. `extra` is an optional appended block.

    NOTE: this does NOT include the date line — that is volatile and is appended
    fresh on every turn by the loop via `with_today()`, so it always reflects the
    current day without freezing it at construction time.
    """
    layers = [load_soul(soul_path, explicit=soul)]
    if tool_guidance:
        layers.append(TOOL_GUIDANCE)
    if _meaningful(extra):
        layers.append(extra.strip())
    return "\n\n".join(layers)


def skills_block(skills: list, limit: int = 30) -> str:
    """Render the per-user skill catalog for injection into the system prompt.

    Only name + summary are listed — NOT the bodies (steps). The model loads a
    skill's body on demand via GetSkill. Returns '' when the user has no skills,
    so the caller injects nothing. If the catalog exceeds `limit`, only the
    first `limit` are shown and the model is pointed at SearchSkills for the
    rest (the hybrid: small catalog in-prompt, long tail searched on demand).

    Placed AFTER the stable global prompt layers so the shared identity / tool
    guidance prefix stays cacheable across users; only this small per-user block
    (plus the volatile date) follows it.
    """
    if not skills:
        return ""
    shown = skills[:limit]
    lines = [f"- {s.name}: {s.summary}" for s in shown]
    body = "\n".join(lines)
    if len(skills) > limit:
        body += (f"\n…and {len(skills) - limit} more not shown — use "
                 "SearchSkills(query) to find them.")
    return (
        "# Your saved skills\n"
        "Reusable procedures saved for this user. If one fits the task, call "
        "GetSkill(name) to load its full steps, then follow them. Only the name "
        "and one-line summary are shown here; the steps live in the body.\n"
        f"{body}"
    )


def today_line(today: Optional[_date] = None) -> str:
    """The volatile date layer, at day granularity (ISO 8601)."""
    d = today or _date.today()
    return f"Today's date is {d.isoformat()}."


def with_today(system_prompt: str, today: Optional[_date] = None) -> str:
    """Append the current date as the final layer of the system prompt."""
    return f"{system_prompt}\n\n{today_line(today)}"
