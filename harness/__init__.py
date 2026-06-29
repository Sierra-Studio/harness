"""A pluggable, multi-tenant LLM agent harness.

Components: Memory, Skills, MCP, Tools, Loop, OpenRouter provider, Observability.
See plano-implementacao.html for the design.
"""

__version__ = "1.0.0"

from .app import Harness

__all__ = ["Harness"]
