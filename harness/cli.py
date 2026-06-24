"""Command-line entrypoint.

    uv run harness init-db     # create schema in DATABASE_URL
    uv run harness chat        # interactive session (uses OpenRouter if configured)

(or without uv:  python -m harness.cli init-db | chat)
"""
from __future__ import annotations

import sys
from pathlib import Path

from .app import Harness
from .config import load_config


def init_db() -> int:
    cfg = load_config()
    if not cfg.database_url:
        print("DATABASE_URL is not set.")
        return 1
    import psycopg

    schema = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()
    schema = schema.replace("{{EMBEDDING_DIM}}", str(cfg.embedding_dim))
    with psycopg.connect(cfg.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(schema)
    print(f"Schema applied to {cfg.database_url} (embedding_dim={cfg.embedding_dim}).")
    return 0


def chat() -> int:
    cfg = load_config()
    h = Harness(cfg, echo=True)
    backend = "Postgres" if cfg.database_url else "in-memory"
    provider = "OpenRouter" if cfg.openrouter_api_key else "FakeProvider (offline)"
    print(f"Harness ready · repo={backend} · provider={provider} · model={cfg.model}")
    external_id = input("user id: ").strip() or "demo-user"
    session = h.start_session(external_id)
    print(f"session {session.id} (context_window={session.context_window}, "
          f"budget={session.token_budget})")
    try:
        while True:
            msg = input("\nyou> ").strip()
            if msg in {"exit", "quit"}:
                break
            res = h.run_turn(session, msg)
            print(f"\nassistant> {res.text}")
            print(f"  [status={res.status} steps={res.steps} "
                  f"tokens_spent={res.tokens_spent}]")
    except (EOFError, KeyboardInterrupt):
        pass
    created = h.close_session(session)
    if created:
        print(f"\nInduced skills: {created}")
    print("session closed.")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "chat"
    if cmd == "init-db":
        return init_db()
    if cmd == "chat":
        return chat()
    print(__doc__)
    return 1


def entry() -> int:
    """Console-script entry point (`harness ...`)."""
    return main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
