"""Facade that wires all components from a Config. Swap any piece here."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal

from ..llm import Provider
from ..mcp import HttpMcpClient, McpClient
from ..memory import Memory, RepositorySkills, Skills, build_system_prompt
from ..observability import Observer, Tracer
from ..persistence import InMemoryRepository, Repository
from ..settings import Config
from ..tools import (
    LocalSubprocessSandbox,
    ProviderHost,
    SandboxBackend,
    Tool,
    ToolProvider,
    ToolRegistry,
)
from .loop import AgentLoop, Hook, LoopEvent, TurnResult
from .permissions import Permissions


class Harness:
    def __init__(
        self,
        cfg: Config,
        *,
        provider: Provider,
        repo: Repository | None = None,
        system_prompt: str = "",
        persona: str = "",
        echo: bool = False,
        sandbox: SandboxBackend | None = None,
        mcp_clients: dict | None = None,
        tools: Iterable[Tool | ToolProvider] | bool | None = None,
        hooks: list[Hook] | None = None,
        skills: Skills | None = None,
        tracer: Tracer | None = None,
        permissions: Permissions | None = None,
    ):
        """Wire a harness from a Config, overriding any component.

        Args:
            cfg: Configuration. Required — the library never reads env vars or
                builds one for you; pass `Config()` for pure defaults or
                `Config.from_env()` if your application opts into that.
            provider: LLM Provider. Required — there is no sensible generic
                default for "which AI backend to talk to"; construct one
                directly (`AzureFoundryProvider(cfg.provider)`, ...) or look
                one up via `build_provider(cfg)` / `ProviderRegistry`.
            repo: Repository backend. Optional with a STATIC default —
                `InMemoryRepository()` — never one derived by inspecting
                `cfg.database_url`. Pass `PostgresRepository(dsn)` explicitly
                for durable storage.
            system_prompt: Explicit system prompt. Overrides the assembled
                persona + per-tool guidance when non-empty.
            persona: Persona/identity text; used to build the prompt when
                `system_prompt` is empty. See `prompt.load_persona` for
                precedence.
            echo: If True, the Observer echoes events to stdout.
            sandbox: SandboxBackend for Bash; defaults to
                `LocalSubprocessSandbox`.
            mcp_clients: Pre-connected MCP clients (name -> client). Usually
                left empty and populated via `add_mcp_stdio` / `add_mcp_http`.
            tools: A single mixed list of `Tool` (inert leaves) and
                `ToolProvider` (capability modules — a sandbox, an MCP server,
                a bundle of domain tools) items, installed in order.
                `None` (the default) means all built-ins (`default_tools()`);
                `True` is the same as `None`; `False` (or `[]`) means nothing
                at all; a list means exactly those. Each `ToolProvider` is `register()`-ed BEFORE the
                prompt is assembled (so its tools/guidance are advertised to
                the model this turn) and `stop()`-ped by `close()`.
            hooks: List of `Hook` objects. Each hook's lifecycle methods
                (`before_turn` / `after_turn` / `before_tool` / `after_tool`)
                fire at the matching point, in list order.
            skills: `Skills` backend for SearchSkills/GetSkill, the per-user
                prompt catalog, and induction. Optional with a STATIC default —
                `RepositorySkills(repo, provider, cfg, observer)`, wired to the
                already-injected `repo`/`provider` — never one derived by
                inspecting `Config`. Pass `skills=NullSkills()` to disable the
                feature entirely (no catalog, no induction) without touching
                `Repository`, or your own `Skills` implementation (e.g. one
                backed by embeddings or an external service).
            tracer: `Tracer` for spans/traces of internal execution (model
                calls, tool calls, memory summarization, skill induction) —
                the observability seam, same DI pattern as everything else.
                Optional with a static default, `NullTracer` (a no-op).
                Bridge to your own system (OpenTelemetry, Datadog, plain
                `logging`, ...) by subclassing `Tracer` and overriding `span`;
                `LoggingTracer` ships as a minimal, dependency-free example.
        """
        self.cfg = cfg
        self.repo = repo or InMemoryRepository()
        self.provider = provider
        self.sandbox = sandbox or LocalSubprocessSandbox(max_output=self.cfg.bash.max_output)
        self.observer = Observer(self.repo, echo=echo, tracer=tracer)
        self.skills = skills or RepositorySkills(self.repo, self.provider, self.cfg, self.observer)
        # The tool set is a single list (default: all built-ins). Include an
        # item to have it, omit it to not — the list IS the selection. Each
        # ToolProvider is registered here, before prompt assembly, so its
        # tools/guidance reach the model this turn.
        self.tools = ToolRegistry(
            self.repo,
            self.sandbox,
            mcp_clients or {},
            config=self.cfg,
            skills=self.skills,
            tools=tools,
            on_provider_error=self._on_provider_error,
        )
        # Permission gate: manual mode asks before each side-effecting tool call.
        # Interfaces (TUI/CLI) install an `asker` and can flip the mode at runtime.
        self.permissions = permissions or Permissions(mode=self.cfg.permissions.mode)
        self.memory = Memory(self.repo, self.provider, self.cfg, self.observer)
        # System prompt: explicit override wins; otherwise assemble the layered
        # persona (PERSONA.md / default identity) plus guidance composed from
        # the active tools' own snippets.
        prompt = system_prompt or build_system_prompt(
            self.cfg.memory.persona_path, persona=persona, tools=self.tools.active_tools()
        )
        self.loop = AgentLoop(
            self.cfg,
            self.repo,
            self.provider,
            self.memory,
            self.tools,
            self.observer,
            prompt,
            self.skills,
            hooks=hooks,
            permissions=self.permissions,
        )

    def _on_provider_error(self, provider: ToolProvider, error: Exception) -> None:
        # Optional providers (e.g. an unreachable sandbox) degrade gracefully;
        # required ones fail construction (ToolRegistry re-raises before this
        # is ever called).
        self.observer.log(
            None,
            None,
            "provider_unavailable",
            {"provider": type(provider).__name__, "error": str(error)},
        )

    # convenience pass-throughs
    def start_session(self, external_id: str, model: str = "", *, session_id: str = ""):
        """Start (or resume) a session for `external_id`.

        Resume `session_id` if it's given and still exists in this Harness's
        repo; otherwise create a new session. The caller supplies the exact id
        it's resuming (from its own cookie/DB row/thread mapping) — this
        performs no lookup beyond checking that one specific id exists.

        SECURITY: this method performs NO authorization/ownership check — it
        does not verify that `session_id` belongs to `external_id`, and it
        never will (Harness has no concept of "the current authenticated
        caller" and can't validate one). Never pass a session_id sourced
        directly from unauthenticated client input (a raw query param, an
        unsigned cookie). Resolve it first through your own access-controlled
        lookup — e.g. a table keyed by your app's own conversation id, scoped
        to the already-authenticated user — and only pass the resulting
        trusted session_id here.
        """
        if session_id:
            existing = self.repo.find_session(session_id)
            if existing is not None:
                return existing
        return self.loop.start_session(external_id, model)

    def run_turn(self, session, message: Any) -> TurnResult:
        return self.loop.run_turn(session, message)

    def run_turn_stream(self, session, message: Any) -> Iterator[LoopEvent]:
        """Yield LoopEvents (text deltas + tool_start/tool_result + final) as the
        turn runs. See AgentLoop.run_turn_stream."""
        return self.loop.run_turn_stream(session, message)

    # ---- runtime controls (change a running Harness without reconstructing it) ----
    def set_persona(self, persona: str = "", system_prompt: str = "") -> None:
        """Rebuild the system prompt in place: `system_prompt` wins if given,
        else the persona/default-identity + per-tool-guidance layering — same
        precedence as the constructor. Call with no args to reset to the
        default identity. Affects every session sharing this Harness, from
        the next turn onward."""
        self.loop.system_prompt = system_prompt or build_system_prompt(
            self.cfg.memory.persona_path, persona=persona, tools=self.tools.active_tools()
        )

    def set_session_model(self, session, model: str) -> None:
        """Switch the model used for `session`'s remaining turns (same
        provider — swapping providers needs a new Provider instance, passed to
        a new Harness). Refreshes `context_window` via the provider so the
        memory budget reflects the new model's real window."""
        session.model = model
        session.context_window = self.provider.model_context_window(model)

    def set_session_budget(self, session, token_budget: int) -> None:
        """Change `session`'s token-spend cap for its remaining turns (0 means
        unlimited). Only affects this session object; new sessions still get
        `cfg.loop.token_budget_per_session`."""
        session.token_budget = max(0, token_budget)

    # ---- stateless / external-history execution mode ----
    def run_stateless_stream(
        self,
        history: list[dict],
        message: Any,
        *,
        external_id: str = "stateless",
        model: str = "",
    ) -> Iterator[LoopEvent]:
        """Run one turn against caller-owned history with no persistence.

        Requires an ephemeral repo (raises otherwise, to avoid silently
        creating orphan rows against a real Postgres-backed harness) — the
        caller must have passed `repo=InMemoryRepository()` (the default) at
        construction. `history` items are `{"role": "user"|"assistant",
        "content": ...}`; anything else is ignored.
        """
        if not isinstance(self.repo, InMemoryRepository):
            raise RuntimeError(
                "run_stateless_stream requires repo=InMemoryRepository() to have been "
                "passed explicitly when constructing this Harness."
            )
        session = self.start_session(external_id, model=model)
        for turn in history:
            role, content = turn.get("role"), turn.get("content")
            if role in ("user", "assistant") and content:
                self.loop.memory.append(session, role, content)
        yield from self.run_turn_stream(session, message)

    def _seed_message(self, session, msg: dict) -> None:
        """Append one caller-supplied message to a stateless session, preserving
        tool-call structure so a suspended turn can be rebuilt faithfully.

        Unlike the plain user/assistant-text seeding in run_stateless_stream,
        this keeps assistant messages that carry `tool_calls` (stored verbatim,
        the same shape the loop records) and `tool` result messages, so
        build_window round-trips them back to the model on resume. `system`
        messages are dropped — the harness prepends its own.
        """
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if content:
                self.loop.memory.append(session, "user", content)
        elif role == "assistant":
            self.loop.memory.append(session, "assistant", msg)
        elif role == "tool":
            self.loop.memory.append(
                session,
                "tool",
                {"tool_call_id": msg.get("tool_call_id"), "content": msg.get("content", "")},
            )

    def resume_stateless_stream(
        self,
        seed_messages: list[dict],
        approved_call: dict,
        *,
        external_id: str = "stateless",
        model: str = "",
    ) -> Iterator[LoopEvent]:
        """Resume a turn that suspended on the permission gate, with no persistence.

        `seed_messages` is the caller-owned window captured at suspension (system
        excluded) — including the assistant message whose tool call is pending.
        `approved_call` is the exact call to run (or a denial). The seeded window
        is rebuilt, the approved call executed verbatim, and the step loop
        continues — deterministically, without re-invoking the model to reproduce
        the tool call. Requires an ephemeral repo, like run_stateless_stream.
        """
        if not isinstance(self.repo, InMemoryRepository):
            raise RuntimeError(
                "resume_stateless_stream requires repo=InMemoryRepository() to have been "
                "passed explicitly when constructing this Harness."
            )
        session = self.start_session(external_id, model=model)
        for msg in seed_messages:
            self._seed_message(session, msg)
        yield from self.loop.resume_turn_stream(session, approved_call)

    def run_stateless(self, history: list[dict], message: Any, **kwargs: Any) -> TurnResult:
        """Collects the final TurnResult, mirroring run_turn/run_turn_stream."""
        result: TurnResult | None = None
        for ev in self.run_stateless_stream(history, message, **kwargs):
            if ev.kind == "final":
                result = ev.result
        assert result is not None
        return result

    def close_session(self, session) -> list[str]:
        self.repo.close_session(session.id)
        self.sandbox.destroy(session.id)
        return self.skills.on_session_closed(session)

    def close(self) -> None:
        """Release harness-lifetime resources: stop every registered provider
        (closing its connections/subprocesses). Call once when the harness is
        retired. Safe to call more than once; provider `stop` should be
        idempotent."""
        for p in self.tools.providers:
            try:
                p.stop()
            except Exception as e:  # teardown must not raise
                self.observer.log(None, None, "provider_stop_error", {"error": str(e)})

    # ---- MCP wiring: index a server's tools and enable dispatch ----
    def _connect_mcp(self, client, transport: str, expose: Literal["index", "direct"]) -> int:
        """Shared MCP hookup, delegating the actual wiring to ProviderHost.add_mcp
        (the single source of truth). `expose="index"` (default) keeps the
        discovery model; `expose="direct"` registers each tool as a first-class
        `McpProxyTool`. Returns the tool count."""
        n = ProviderHost(self.tools, self.repo).add_mcp(client, expose=expose)
        self.observer.log(
            None,
            None,
            "mcp_connected",
            {"server": client.name, "transport": transport, "tools": n, "expose": expose},
        )
        return n

    def add_mcp_stdio(
        self, command: list[str], name: str, *, expose: Literal["index", "direct"] = "index"
    ) -> McpClient:
        """Connect a local stdio MCP server (subprocess). See `_connect_mcp` for
        the `expose` policy."""
        client = McpClient(command, name)
        self._connect_mcp(client, "stdio", expose)
        return client

    def add_mcp_http(
        self,
        url: str,
        name: str = "",
        headers: dict | None = None,
        oauth=None,
        *,
        expose: Literal["index", "direct"] = "index",
    ) -> HttpMcpClient:
        """Connect a remote Streamable-HTTP MCP server.

        Auth options: pass a static bearer via headers
        ({"Authorization": f"Bearer {token}"}), or enable the interactive OAuth
        flow with oauth=True (or an OAuthConfig).

        `expose` selects how the server's tools reach the model: "index" (default)
        via discovery, or "direct" as first-class tools. See `_connect_mcp`.
        """
        client = HttpMcpClient(url, name, headers=headers, oauth=oauth)
        self._connect_mcp(client, "http", expose)
        return client
