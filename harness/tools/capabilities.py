"""Tool providers — the uniform way to compose capabilities into a harness.

A `Tool` is a single leaf extension. A `ToolProvider` is the next level up: a
self-contained *capability module* that owns its own resources (connections,
clients), contributes one or more tools, and tears itself down. Every capability
— a sandbox, an MCP server, a bundle of domain tools — is the SAME kind of thing,
added the SAME way, with the SAME lifecycle:

    Harness(cfg, providers=[
        DomainTools(...),                          # app-supplied bundle
        McpServer(sandbox_url, "sandbox", expose="direct"),
        McpServer(jira_url, "jira", expose="index"),
    ])

Contract:
  * `register(host)` — contribute tools via the `ProviderHost` handed in. Called
    once during harness construction, BEFORE the system prompt is assembled, so a
    provider's tools (and their guidance) are advertised to the model.
  * `stop()` — release resources. Called by `Harness.close()`. Default no-op.

`ProviderHost` is the small, stable surface a provider uses to wire itself in —
it hides the registry/repo internals so providers never poke them directly.
"""

from __future__ import annotations

import abc
from typing import Literal

from ..mcp import HttpMcpClient, McpClient, ingest_server
from ..persistence import Repository
from .builtin import McpProxyTool, Tool, ToolRegistry


class ProviderHost:
    """The surface a `ToolProvider` uses to contribute to a harness.

    Keeps providers decoupled from registry/repo internals: they call these
    methods instead of mutating harness state directly.
    """

    def __init__(self, registry: ToolRegistry, repo: Repository):
        self._registry = registry
        self._repo = repo

    def add_tool(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a single direct tool (its schema is sent to the model)."""
        self._registry.register(tool, replace=replace)

    def add_mcp(self, client, *, expose: Literal["index", "direct"] = "index") -> int:
        """Connect one MCP client and wire its tools per the exposure policy.

        The single source of truth for MCP wiring, shared by `McpServer` and the
        `Harness.add_mcp_*` convenience methods. Always indexes the server (so its
        tools are discoverable and dispatchable by name); with `expose="direct"`
        also registers each as a first-class `McpProxyTool`. Returns the tool count.
        """
        client.start()
        self._registry.mcp_clients[client.name] = client
        n = ingest_server(self._repo, client)
        if expose == "direct":
            for spec in client.list_tools():
                if spec.get("name"):
                    self._registry.register(McpProxyTool(client, spec), replace=True)
        return n


class ToolProvider(abc.ABC):
    """A composable capability module. See module docstring.

    Set `optional = True` on a provider whose failure to register (e.g. an
    unreachable server) should degrade gracefully — the harness logs and skips it
    instead of aborting construction. Default is fail-fast.
    """

    optional: bool = False

    @abc.abstractmethod
    def register(self, host: ProviderHost) -> None: ...

    def stop(self) -> None:  # noqa: B027 - deliberately a no-op default, not abstract
        """Release resources (connections, subprocesses). Default: nothing."""


class ToolBundle(ToolProvider):
    """The simplest provider: contributes a fixed list of already-built tools.

    For stateless tools that need no connection lifecycle — wrap a list so it
    composes uniformly alongside connection-bearing providers. Apps with richer
    needs (lazy clients, config) subclass `ToolProvider` directly instead.
    """

    def __init__(self, tools: list[Tool]):
        self._tools = list(tools)

    def register(self, host: ProviderHost) -> None:
        for t in self._tools:
            host.add_tool(t)


class _McpServerBase(ToolProvider):
    """Shared MCP-provider logic; subclasses only supply the client."""

    def __init__(self, name: str, expose: Literal["index", "direct"], optional: bool):
        self.name = name
        self.expose = expose
        self.optional = optional
        self._client = None

    def _make_client(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def register(self, host: ProviderHost) -> None:
        client = self._make_client()
        host.add_mcp(client, expose=self.expose)
        self._client = client

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
            self._client = None


class McpServer(_McpServerBase):
    """A remote Streamable-HTTP MCP server as a provider.

    `expose="index"` (default) keeps the discovery model; `expose="direct"` sends
    the server's tools to the model as first-class functions.
    """

    def __init__(
        self,
        url: str,
        name: str = "",
        *,
        headers: dict | None = None,
        oauth=None,
        expose: Literal["index", "direct"] = "index",
        optional: bool = False,
    ):
        super().__init__(name or url, expose, optional)
        self._url = url
        self._headers = headers
        self._oauth = oauth

    def _make_client(self) -> HttpMcpClient:
        return HttpMcpClient(self._url, self.name, headers=self._headers, oauth=self._oauth)


class McpStdioServer(_McpServerBase):
    """A local stdio (subprocess) MCP server as a provider."""

    def __init__(
        self,
        command: list[str],
        name: str,
        *,
        expose: Literal["index", "direct"] = "index",
        optional: bool = False,
    ):
        super().__init__(name, expose, optional)
        self._command = command

    def _make_client(self) -> McpClient:
        return McpClient(self._command, self.name)
