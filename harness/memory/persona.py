"""System-prompt assembly (the "Persona").

Inspired by Hermes' SOUL.md + layered prompt. The prompt is built from ordered
layers so the stable identity sits first (cache-friendly) and per-call guidance
follows:

    [ persona / identity ]   <- PERSONA.md if present & non-empty, else DEFAULT_IDENTITY
    [ tool guidance       ]  <- composed from the ACTIVE tools' own guidance snippets
    [ extra message       ]  <- optional caller-supplied system_message

The tool-guidance layer is not hardcoded: each Tool carries its own `guidance`
snippet and `compose_tool_guidance` assembles only the active ones, so the prompt
never mentions a tool that isn't present.

PERSONA.md is loaded fresh; delete it (or its content) to fall back to the
default persona. Comment-only files (HTML comments) count as empty. SOUL.md is
still read as a deprecated fallback filename for one release (logged as a
deprecation warning) if no PERSONA.md is found.
"""

from __future__ import annotations

import re
import warnings
from datetime import date as _date
from pathlib import Path

DEFAULT_IDENTITY = (
    "You are a capable AI assistant running inside a harness. You are helpful, "
    "direct, and honest. You assist with answering questions, writing and editing "
    "code, analyzing information, and executing actions through your tools. You "
    "communicate clearly, admit uncertainty when appropriate, and prioritize being "
    "genuinely useful over being verbose. Be targeted and efficient."
)

# Tool-agnostic framing wrapped around the active tools' own guidance snippets.
_GUIDANCE_HEADER = (
    "# Your tools\n"
    "You have the tools listed below, plus external tools you can discover on "
    "demand. On every action, prefer a saved skill that fits, then a specialized "
    "external tool, and reach for Bash as the universal fallback when no specific "
    "tool fits but the operating system can do the job."
)
_GUIDANCE_CLOSING = (
    "When a tool or command fails and blocks the real path, say so honestly and try "
    "an alternative. Never substitute plausible-looking fabricated results for "
    "output you could not actually produce."
)


def compose_tool_guidance(tools) -> str:
    """Assemble the tool-guidance layer from the active tools' `guidance` snippets.

    Each tool contributes its own block (in the given order); tools with empty
    guidance contribute nothing. Returns '' when no tool has guidance, so the
    caller injects no guidance layer at all.
    """
    blocks = [g for t in tools if (g := getattr(t, "guidance", "").strip())]
    if not blocks:
        return ""
    return "\n".join([_GUIDANCE_HEADER, *blocks, _GUIDANCE_CLOSING])


_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _meaningful(text: str) -> str:
    """Strip HTML comments and whitespace; return '' if nothing remains."""
    stripped = _HTML_COMMENT.sub("", text or "").strip()
    return stripped


def load_persona(persona_path: str = "", *, explicit: str = "") -> str:
    """Resolve the persona/identity text.

    Precedence: explicit string > PERSONA.md at persona_path > PERSONA.md in
    CWD > SOUL.md at persona_path (deprecated) > SOUL.md in CWD (deprecated) >
    DEFAULT_IDENTITY. A comment-only or empty file falls through to the next
    candidate.
    """
    if _meaningful(explicit):
        return _meaningful(explicit)

    candidates = []
    if persona_path:
        candidates.append(Path(persona_path))
    candidates.append(Path.cwd() / "PERSONA.md")

    for path in candidates:
        try:
            if path.is_file():
                content = _meaningful(path.read_text(encoding="utf-8"))
                if content:
                    return content
        except OSError:
            continue

    # Deprecated fallback filename — dropped in a future release.
    deprecated = []
    if persona_path:
        deprecated.append(Path(persona_path).with_name("SOUL.md"))
    deprecated.append(Path.cwd() / "SOUL.md")
    for path in deprecated:
        try:
            if path.is_file():
                content = _meaningful(path.read_text(encoding="utf-8"))
                if content:
                    warnings.warn(
                        f"{path} uses the deprecated SOUL.md filename; rename it to "
                        "PERSONA.md (SOUL.md support will be removed in a future release).",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return content
        except OSError:
            continue

    return DEFAULT_IDENTITY


def build_system_prompt(
    persona_path: str = "", *, persona: str = "", tools=None, extra: str = ""
) -> str:
    """Assemble the layered system prompt. `extra` is an optional appended block.

    When `tools` is given, a guidance layer is composed from those tools'
    `guidance` snippets (see `compose_tool_guidance`); when it is None or none of
    the tools carry guidance, no guidance layer is added.

    NOTE: this does NOT include the date line — that is volatile and is appended
    fresh on every turn by the loop via `with_today()`, so it always reflects the
    current day without freezing it at construction time.
    """
    layers = [load_persona(persona_path, explicit=persona)]
    guidance = compose_tool_guidance(tools) if tools else ""
    if guidance:
        layers.append(guidance)
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
        body += (
            f"\n…and {len(skills) - limit} more not shown — use SearchSkills(query) to find them."
        )
    return (
        "# Your saved skills\n"
        "Reusable procedures saved for this user. If one fits the task, call "
        "GetSkill(name) to load its full steps, then follow them. Only the name "
        "and one-line summary are shown here; the steps live in the body.\n"
        f"{body}"
    )


def today_line(today: _date | None = None) -> str:
    """The volatile date layer, at day granularity (ISO 8601)."""
    d = today or _date.today()
    return f"Today's date is {d.isoformat()}."


def with_today(system_prompt: str, today: _date | None = None) -> str:
    """Append the current date as the final layer of the system prompt."""
    return f"{system_prompt}\n\n{today_line(today)}"
