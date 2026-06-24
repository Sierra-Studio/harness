"""Facade that wires all components from a Config. Swap any piece here."""
from __future__ import annotations

from .config import Config, load_config
from .embeddings import Embedder
from .loop import AgentLoop
from .mcp_client import HttpMcpClient, McpClient, ingest_server
from .memory import Memory
from .observer import Observer
from .provider import Provider, build_provider
from .repository import Repository, build_repository
from .sandbox import LocalSubprocessSandbox, SandboxBackend
from .skills import SkillInducer
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a capable assistant running inside a harness. You have four built-in "
    "tools: SearchTools (find external tools by description), GetTools (fetch a "
    "tool's schema), GetSkills (recall the user's saved procedures), and Bash (run "
    "shell commands in your sandbox). Prefer SearchTools over guessing tool names. "
    "Be concise."
)


class Harness:
    def __init__(self, cfg: Config | None = None, *, system_prompt: str = "",
                 echo: bool = False, repo: Repository | None = None,
                 provider: Provider | None = None,
                 sandbox: SandboxBackend | None = None,
                 mcp_clients: dict | None = None):
        self.cfg = cfg or load_config()
        self.repo = repo or build_repository(self.cfg)
        self.provider = provider or build_provider(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.sandbox = sandbox or LocalSubprocessSandbox()
        self.observer = Observer(self.repo, echo=echo)
        self.tools = ToolRegistry(self.repo, self.embedder, self.sandbox,
                                  mcp_clients or {})
        self.memory = Memory(self.repo, self.provider, self.cfg, self.observer)
        self.inducer = SkillInducer(self.repo, self.provider, self.embedder,
                                    self.cfg, self.observer)
        self.loop = AgentLoop(self.cfg, self.repo, self.provider, self.memory,
                              self.tools, self.observer,
                              system_prompt or DEFAULT_SYSTEM_PROMPT)

    # convenience pass-throughs
    def start_session(self, external_id: str, model: str = ""):
        return self.loop.start_session(external_id, model)

    def run_turn(self, session, message: str):
        return self.loop.run_turn(session, message)

    def close_session(self, session) -> list[str]:
        self.repo.close_session(session.id)
        self.sandbox.destroy(session.id)
        return self.inducer.on_session_closed(session)

    # ---- MCP wiring: index a server's tools and enable dispatch ----
    def add_mcp_stdio(self, command: list[str], name: str) -> McpClient:
        """Connect a local stdio MCP server (subprocess)."""
        client = McpClient(command, name)
        client.start()
        n = ingest_server(self.repo, client)
        self.tools.mcp_clients[client.name] = client
        self.observer.log(None, None, "mcp_connected",
                          {"server": client.name, "transport": "stdio", "tools": n})
        return client

    def add_mcp_http(self, url: str, name: str = "",
                     headers: dict | None = None) -> HttpMcpClient:
        """Connect a remote Streamable-HTTP MCP server.

        Pass headers for auth, e.g. {"Authorization": f"Bearer {token}"}.
        """
        client = HttpMcpClient(url, name, headers=headers)
        client.start()
        n = ingest_server(self.repo, client)
        self.tools.mcp_clients[client.name] = client
        self.observer.log(None, None, "mcp_connected",
                          {"server": client.name, "transport": "http", "tools": n})
        return client
