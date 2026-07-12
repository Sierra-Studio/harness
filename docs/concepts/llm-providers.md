# Providers (LLM)

`harness/llm/provider.py` defines the `Provider` contract — the harness
depends only on this, never on a concrete backend:

```python
class Provider(abc.ABC):
    def model_context_window(self, model: str) -> int: ...
    def complete(self, model, messages, tools=None) -> ModelResult: ...
    def stream(self, model, messages, tools=None) -> Generator[str, None, ModelResult]: ...

    # helpers built on top of complete()
    def summarize(self, model, prev_summary, messages) -> str: ...
    def classify_subject(self, model, messages) -> str: ...
    def induce_skills(self, model, signals) -> list[dict]: ...
```

`stream()` has a non-streaming default (one delta = the whole text), so any
`Provider` works without implementing real streaming.

!!! note "Not to be confused with tool providers"
    This is the **LLM** provider. `harness.tools` (`ToolProvider`) is a
    different concept — composing capabilities (MCP servers, tool bundles)
    into a harness. See [Tools & Providers](tools-and-providers.md).

## Implementations

- **`OpenRouterProvider`** — model context window from `/models`, chat
  completions with tool calling, real `usage` recorded per turn.
- **`AzureFoundryProvider`** — Azure AI Foundry's OpenAI-compatible `/openai/v1`
  endpoint. Auth precedence: a static `AZURE_AI_API_KEY` if set, otherwise a
  Microsoft Entra ID bearer token via `azure-identity` (managed identity in
  Azure, `DefaultAzureCredential` for local dev). Needs the `azure` extra:
  `pip install 'harness[azure]'`.
- **`FakeProvider`** — deterministic, network-free provider for the offline
  demo and tests. Drive it with a queued script of responses
  (`provider.queue(content=..., tool_calls=...)`); with no queue it echoes
  `"OK: {last message}"`.

Both `OpenRouterProvider` and `AzureFoundryProvider` share request/response and
SSE-streaming logic via `OpenAICompatibleProvider` — they differ only in
transport/auth (`_client`), extra query params (`_params`), and how they
report a model's context window.

## Selecting a provider

`build_provider(cfg)` is an **explicit-name lookup only** — it performs no
detection itself. The caller supplies `cfg.provider.name` (from a config file,
CLI flag, or per-tenant setting); it resolves that name through
`ProviderRegistry`.

```python
from harness.llm import build_provider
provider = build_provider(cfg)   # requires cfg.provider.name to be set
```

`detect_provider(cfg)` is a separate, opt-in convenience for demos/CLIs that
don't want to manage `cfg.provider.name`: Azure if `azure_endpoint` is set,
else OpenRouter if `openrouter_api_key` is set, else the offline
`FakeProvider`. It is **not** called automatically by `Harness` or
`build_provider` — `harness/interfaces/cli.py` is the one place that opts into it.

## Provider registry

`harness/llm/registry.py`'s `ProviderRegistry` / `@register_provider` is a library
utility the application may choose to use — nothing here is invoked
automatically by `Harness`. Built-in providers self-register at import time
(`"openrouter"`, `"azure"`, `"fake"`); register your own to make it selectable
by name (e.g. picked by an env var or per-tenant setting).

```python
from harness.llm import register_provider

@register_provider("my-gateway")
class MyProvider(Provider):
    ...
```
