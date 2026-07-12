"""Explicit, name-keyed provider registry.

This is a library UTILITY the application may choose to use — nothing here is
invoked automatically by `Harness`. Register a factory under a name (built-ins
self-register at import time via `@register_provider`), then build one by
name when the application wants data-driven provider selection (e.g. picked
by an env var or per-tenant setting) rather than fixed in code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ProviderConfig
    from .provider import Provider


class ProviderRegistry:
    _factories: dict[str, Callable[[ProviderConfig], Provider]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[ProviderConfig], Provider]) -> None:
        cls._factories[name] = factory

    @classmethod
    def build(cls, name: str, cfg: ProviderConfig) -> Provider:
        try:
            factory = cls._factories[name]
        except KeyError:
            raise ValueError(
                f"Unknown provider {name!r}; registered: {list(cls._factories)}"
            ) from None
        return factory(cfg)


def register_provider(name: str):
    """Class decorator: registers `cls` under `name` at import time."""

    def deco(cls):
        ProviderRegistry.register(name, cls)
        return cls

    return deco


__all__ = ["ProviderRegistry", "register_provider"]
