"""Memory, Persona, and Skills — the pieces that build and manage conversation context."""

from __future__ import annotations

from .persona import (
    DEFAULT_IDENTITY,
    build_system_prompt,
    compose_tool_guidance,
    load_persona,
    skills_block,
    today_line,
    with_today,
)
from .skills import NullSkills, RepositorySkills, Skills
from .window import Memory

__all__ = [
    "Memory",
    "DEFAULT_IDENTITY",
    "build_system_prompt",
    "compose_tool_guidance",
    "load_persona",
    "skills_block",
    "today_line",
    "with_today",
    "Skills",
    "NullSkills",
    "RepositorySkills",
]
