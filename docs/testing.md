# Testing

```bash
uv run pytest -q
```

## Test files

| File | Covers |
|---|---|
| `tests/test_core.py` | core loop, memory, tools, providers |
| `tests/test_providers.py` | `ToolProvider` / `ProviderHost` composition, MCP wiring |
| `tests/test_redesign.py` | config/settings split, provider registry |
| `tests/test_skills_and_tracer.py` | Skills and Tracer dependency-injection seams |

All tests force an in-memory repository (`database_url=""`) so they never
depend on ambient `.env`/`DATABASE_URL`, and use `FakeProvider` so they never
make network calls.

## Writing a test

Use `harness.testing.offline_harness()` for a fully-wired, network-free
`Harness`, or build one directly for more control:

```python
import dataclasses
from harness.core import Harness
from harness.settings import Config
from harness.llm import FakeProvider

def make_harness(provider=None, **overrides):
    cfg = dataclasses.replace(Config(), database_url="", **overrides)
    return Harness(cfg, system_prompt="sys.", provider=provider or FakeProvider(context_window=4000))
```

`FakeProvider` is deterministic and network-free: queue scripted responses with
`provider.queue(content=..., tool_calls=...)`, or let it fall through to its
default echo (`"OK: {last message}"`) when the queue is empty.

## Status / notes

- Validated end-to-end against **real Postgres + pgvector** (repository,
  full-text `SearchTools`, summarization, checkpoints, induction, token
  accounting).
- OpenRouter integration verified up to billing: the harness authenticates,
  pulls the model context window, and forms valid chat requests. A live
  completion needs account credits (a `402 Payment Required` means no credit).
