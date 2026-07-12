# Changelog

## 2.1.0

Extends the explicit-DI redesign from 2.0.0 to two more components: Skills and
observability/tracing. Both follow the same rule as `repo=`/`sandbox=`: an
optional constructor kwarg with a static, predictable default — never
something `Harness` selects by inspecting `Config`.

### Added

- **`Harness(skills=...)`** — a new `harness.skills.Skills` abstract seam for
  SearchSkills/GetSkill, the per-user prompt catalog, and induction. Static
  default: `RepositorySkills(repo, provider, cfg, observer)` (today's built-in
  behavior — persists through the injected `Repository`, induces via the
  injected `Provider` on the existing cadence — now just behind a swappable
  interface). Pass `skills=NullSkills()` to disable the feature entirely
  (empty catalog, no induction) without touching `Repository`, or your own
  `Skills` implementation (e.g. embeddings-backed, or an external service).
  `harness.skills.SkillInducer` is gone — its logic moved into
  `RepositorySkills.on_session_closed`.
- **`Harness(tracer=...)`** — a new `harness.observer.Tracer` seam for
  spans/traces of internal execution (model calls, tool calls, memory
  summarization, skill induction), bridging into OpenTelemetry, Datadog,
  plain `logging`, or anything else. Static default: `NullTracer` (a no-op,
  identical to today's behavior). `Observer(repo, tracer=...)` takes the same
  kwarg directly. `LoggingTracer` ships as a minimal, dependency-free example
  (writes each span to Python's `logging` module) — subclass `Tracer` and
  override `span` to bridge to a real backend. `Observer.timed()` opens one
  span per call (also feeding the existing `repo.add_step_log` persistence,
  unchanged); one-off `Observer.log()` calls (outside `timed()`) forward to
  `tracer.event()` instead, so nothing is traced twice.

### Changed (advanced/internal — no change if you only construct `Harness`)

- `ToolContext` gained a `skills: Skills` field; `ToolRegistry.__init__` gained
  a required `skills` kwarg. Only affects code that constructs `ToolRegistry`
  directly (bypassing `Harness`) — e.g. tests.
- `AgentLoop.__init__` gained a required `skills` positional param (after
  `system_prompt`). Same scope as above: only direct `AgentLoop` construction
  is affected.
- `SearchSkills`/`GetSkill` tools now read through `ctx.skills` instead of
  `ctx.repo.search_skills`/`ctx.repo.list_skills`/`ctx.repo.get_skill`
  directly — behavior is identical under the default `RepositorySkills`
  (which just delegates to those same `Repository` methods).

## 2.0.0

### Breaking changes

- **`Harness.__init__` requires `cfg` and `provider` explicitly.** The library
  no longer reads env vars or auto-selects an implementation: `cfg or
  load_config()` and `provider or build_provider(cfg)` are gone. `repo` keeps
  a static default (`InMemoryRepository()`), same pattern as `sandbox`.
- **`Config` is now composable, not flat.** `harness.config.Config` moved to
  `harness.settings.Config`, split into `ProviderConfig`, `LoopConfig`,
  `MemoryConfig`, `BashConfig`. Flat attribute access breaks:
  `cfg.max_steps` -> `cfg.loop.max_steps`, `cfg.bash_timeout` ->
  `cfg.bash.timeout`, `cfg.azure_endpoint` -> `cfg.provider.azure_endpoint`,
  etc. `harness.config` still re-exports everything for the import path.
- **`Config()` no longer reads the environment.** All fields are static
  literal defaults. Use the new, explicit `Config.from_env()` (and each
  sub-config's own `from_env()`) when your application wants that — never
  called automatically by `Harness` or by `Config`'s own defaults.
  `load_config()` is removed.
- **`soul` renamed to `persona`** everywhere: the `Harness`/
  `build_system_prompt` kwarg, `load_soul` -> `load_persona`,
  `cfg.soul_path` -> `cfg.memory.persona_path`. `PERSONA.md` is the new
  convention filename; `SOUL.md` is read as a deprecated fallback for one
  release (logs a `DeprecationWarning`), then will be dropped.
- **`tools=` and `providers=` merged into a single `tools=` parameter.**
  `Harness(providers=[...])` is gone — pass `ToolProvider` instances directly
  in the `tools=` list alongside plain `Tool` instances; `ToolRegistry`
  dispatches each by type. `ToolBundle` remains as optional sugar for
  grouping tools, not the only way to mix them in.
- **Providers now take a `ProviderConfig`, not the full `Config`.**
  `OpenRouterProvider(cfg)` / `AzureFoundryProvider(cfg)` now expect
  `cfg.provider`, not `cfg`.
- **`build_provider(cfg)` is an explicit-name-only lookup.** It requires
  `cfg.provider.name` to be set and raises otherwise — no more sniffing
  `azure_endpoint`/`openrouter_api_key` to guess. Use the new
  `detect_provider(cfg)` for that old heuristic (opt-in, not auto-invoked),
  or register your own provider via `@register_provider("name")` /
  `ProviderRegistry`.

### Added

- `LoopConfig.max_tool_calls_per_step` / `max_tool_calls_per_turn` (0 =
  unlimited). Overflowing a step truncates and logs
  `tool_calls_truncated`; overflowing a turn stops with the new
  `TurnResult.status == "tool_limit_exhausted"`.
- `harness.registry.ProviderRegistry` / `@register_provider` — explicit,
  name-keyed provider registration for third-party/custom providers.
- `Repository.find_session(session_id) -> Session | None` and
  `Harness.start_session(external_id, session_id=...)` to resume an existing
  session instead of always creating a new one. Performs no ownership check
  — see the docstring for the security note.
- `Harness.run_stateless_stream` / `run_stateless` — a first-class,
  documented mode for callers who own conversation history themselves
  (e.g. a request-scoped Postgres-backed app) and want a single stateless
  turn against caller-supplied history with an ephemeral
  `InMemoryRepository()`.
- `harness.testing.offline_harness()` — a convenience factory
  (`Config()` + `FakeProvider()`) for demos/tests/quickstarts.
- `detect_provider(cfg)` — opt-in env-sniffing convenience for CLIs/demos
  that don't want to manage `cfg.provider.name` themselves.
