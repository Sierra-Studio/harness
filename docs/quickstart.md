# Quickstart

Requires [uv](https://docs.astral.sh/uv/). One-time setup:

```bash
uv sync --extra dev    # creates .venv and installs deps from pyproject.toml
```

## 1. Offline demo (no DB, no API key)

Runs the whole harness with an in-memory repo and a scripted fake provider:

```bash
uv run python -c "
from harness.testing import offline_harness
h = offline_harness()
session = h.start_session('u1')
for ev in h.run_turn_stream(session, 'hello'):
    print(ev)
"
```

`offline_harness()` is a convenience factory (`Config()` + `FakeProvider()`) for
demos, tests, and quickstarts — `Harness` itself never defaults `cfg`/`provider`.

## 2. Tests

```bash
uv run pytest -q
```

## 3. Real run (OpenRouter or Azure + Postgres)

```bash
cp .env.example .env          # set OPENROUTER_API_KEY (needs credits)
docker compose up -d          # Postgres + pgvector
uv run harness init-db        # apply schema.sql
uv run harness chat           # interactive session
```

With `DATABASE_URL` unset the harness uses the in-memory repo; with no LLM
provider configured it uses the offline `FakeProvider`. Mix and match, e.g. force
offline + in-memory:

```bash
DATABASE_URL="" OPENROUTER_API_KEY="" uv run harness chat
```

!!! note "Not using uv?"
    The core still runs with plain `python` (e.g. `python -m harness.interfaces.cli chat`);
    install deps however you like — they're declared in `pyproject.toml`.

## Swapping pieces

`app.py` is the single wiring point:

```python
Harness(cfg, provider=MyProvider(...))          # different LLM gateway
Harness(cfg, sandbox=FirecrackerSandbox(...))   # kernel-isolated Bash
Harness(cfg, repo=PostgresRepository(dsn))      # forced backend
```

See [Configuration](configuration.md) for `Config`, and
[Concepts](concepts/loop-and-memory.md) for how each swappable piece behaves.
