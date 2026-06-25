"""Dataclasses mirroring the Postgres schema."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class User:
    id: str
    external_id: str


@dataclass
class Session:
    id: str
    user_id: str
    model: str
    context_window: int
    token_budget: int
    tokens_spent: int = 0
    status: str = "open"  # open | closed | budget_exhausted


@dataclass
class Turn:
    id: str
    session_id: str
    user_id: str
    idx: int
    role: str  # user | system | assistant | tool
    content: Any
    token_count: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    in_window: bool = True
    created_at: Optional[datetime] = None


@dataclass
class Summary:
    id: str
    session_id: str
    parent_id: Optional[str]
    content: str
    token_count: int
    covers_until: int


@dataclass
class Skill:
    id: str
    user_id: str
    name: str
    summary: str
    body: str
    origin: str = "induced"  # authored | induced


@dataclass
class ToolSpec:
    id: str
    mcp_server: str
    name: str
    description: str
    input_schema: dict
    enabled: bool = True
