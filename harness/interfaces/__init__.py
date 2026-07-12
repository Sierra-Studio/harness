"""Application-facing entry points: the CLI and the SSE HTTP server.

These are applications, not library-internal code — they're the layer that
opts into `Config.from_env()`, `detect_provider()`, and `build_repository()`.
`Harness` itself never does any of that on its own.
"""

from __future__ import annotations

from .cli import entry
from .server import build_harness, serve

__all__ = ["entry", "serve", "build_harness"]
