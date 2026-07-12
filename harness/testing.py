"""Convenience factory for demos/tests/quickstarts.

`Harness` itself never defaults `cfg`/`provider` — this module is a small,
separate opt-in helper for the common "just give me a fully wired offline
Harness" case, not a hidden default on the constructor's argument path.
"""

from __future__ import annotations

from .core import Harness
from .llm import FakeProvider
from .settings import Config


def offline_harness(**overrides) -> Harness:
    """Fully wired Harness for demos/tests: Config(), FakeProvider() — repo
    defaults to InMemoryRepository() on its own (no override needed).

    A convenience factory, not a hidden default — callers who want explicit
    control just construct Harness(...) directly.
    """
    cfg = overrides.pop("cfg", Config())
    provider = overrides.pop(
        "provider", FakeProvider(context_window=cfg.provider.default_context_window)
    )
    return Harness(cfg, provider=provider, **overrides)


__all__ = ["offline_harness"]
