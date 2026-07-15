"""Shared test helpers.

Importable as `from helpers import ...` because pytest prepends each test
file's directory (tests/) to sys.path. Centralizes the offline-Harness
factory that every test file used to reimplement.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.settings import Config, LoopConfig, MemoryConfig

_UNSET = object()


def make_cfg(*, loop: LoopConfig | None = None, memory: MemoryConfig | None = None, **overrides) -> Config:
    """Config for tests: forces the in-memory repo (database_url="") so tests
    never depend on ambient .env / DATABASE_URL."""
    return dataclasses.replace(
        Config(),
        database_url="",
        loop=loop or LoopConfig(),
        memory=memory or MemoryConfig(),
        **overrides,
    )


def make_harness(
    provider=None,
    *,
    loop: LoopConfig | None = None,
    memory: MemoryConfig | None = None,
    system_prompt: str = "sys.",
    tools: Any = _UNSET,
    hooks: Any = _UNSET,
    **cfg_overrides,
) -> Harness:
    """Fully wired offline Harness: in-memory repo + FakeProvider by default.

    `tools`/`hooks` are forwarded to Harness only when given, so the
    constructor's own defaults (all built-ins, no hooks) stay in charge.
    """
    kwargs: dict = {}
    if tools is not _UNSET:
        kwargs["tools"] = tools
    if hooks is not _UNSET:
        kwargs["hooks"] = hooks
    return Harness(
        make_cfg(loop=loop, memory=memory, **cfg_overrides),
        system_prompt=system_prompt,
        provider=provider or FakeProvider(context_window=4000),
        **kwargs,
    )


def drain(gen):
    """Consume a stream() generator: return (deltas, final ModelResult)."""
    deltas = []
    while True:
        try:
            deltas.append(next(gen))
        except StopIteration as stop:
            return deltas, stop.value
