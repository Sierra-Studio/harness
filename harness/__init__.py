"""A pluggable, multi-tenant LLM agent harness.

Components: Memory, Skills, MCP, Tools, Loop, OpenRouter provider, Observability.
"""

__version__ = "2.1.0"

from .core import (
    Harness,
    Hook,
    TurnResult,
    after_tool,
    after_turn,
    before_tool,
    before_turn,
)
from .llm import ProviderRegistry, register_provider
from .memory import NullSkills, RepositorySkills, Skills
from .observability import LoggingTracer, NullTracer, Observer, Tracer
from .settings import BashConfig, Config, LoopConfig, MemoryConfig, ProviderConfig
from .tools import (
    Bash,
    CallTool,
    GetSkill,
    GetTools,
    McpServer,
    McpStdioServer,
    ProviderHost,
    RenderUI,
    SearchSkills,
    SearchTools,
    Tool,
    ToolBundle,
    ToolContext,
    ToolProvider,
    default_tools,
    make_tool,
    tool,
)

__all__ = [
    "Harness",
    "Hook",
    "TurnResult",
    "before_turn",
    "after_turn",
    "before_tool",
    "after_tool",
    "Config",
    "ProviderConfig",
    "LoopConfig",
    "MemoryConfig",
    "BashConfig",
    "ProviderRegistry",
    "register_provider",
    "Tool",
    "ToolContext",
    "default_tools",
    "make_tool",
    "tool",
    "SearchTools",
    "GetTools",
    "CallTool",
    "SearchSkills",
    "GetSkill",
    "Bash",
    "RenderUI",
    "ToolProvider",
    "ToolBundle",
    "McpServer",
    "McpStdioServer",
    "ProviderHost",
    "Skills",
    "NullSkills",
    "RepositorySkills",
    "Observer",
    "Tracer",
    "NullTracer",
    "LoggingTracer",
]
