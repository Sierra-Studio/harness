"""Persistence layer: the Repository contract plus in-memory and Postgres(pgvector) implementations."""

from __future__ import annotations

from .repository import (
    InMemoryRepository,
    PostgresRepository,
    Repository,
    SQLiteRepository,
    build_repository,
)

__all__ = [
    "Repository",
    "InMemoryRepository",
    "SQLiteRepository",
    "PostgresRepository",
    "build_repository",
]
