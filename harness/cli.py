"""Command-line entrypoint.

    uv run harness init-db     # create schema in DATABASE_URL
    uv run harness chat        # interactive session (uses OpenRouter if configured)
    uv run harness add-skill <user_id> <name> <summary> [body]
                               # author a skill (body read from stdin if omitted)
    uv run harness list-skills <user_id>

(or without uv:  python -m harness.cli init-db | chat | ...)
"""
from __future__ import annotations

import sys
from pathlib import Path

from .app import Harness
from .config import load_config, mcp_http_servers


def init_db() -> int:
    cfg = load_config()
    if not cfg.database_url:
        print("DATABASE_URL is not set.")
        return 1
    import psycopg

    schema = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()
    with psycopg.connect(cfg.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(schema)
    print(f"Schema applied to {cfg.database_url}.")
    return 0


def add_skill(argv: list[str]) -> int:
    """author a skill: add-skill <user_id> <name> <summary> [body]
    If body is omitted, it is read from stdin (so you can pipe or heredoc it)."""
    if len(argv) < 5:
        print("usage: harness add-skill <user_id> <name> <summary> [body]")
        return 1
    user_id, name, summary = argv[2], argv[3], argv[4]
    body = argv[5] if len(argv) > 5 else sys.stdin.read()
    if not body.strip():
        print("ERROR: empty body (provide as arg or via stdin).")
        return 1
    h = Harness(load_config())
    uid = h.repo.get_or_create_user(user_id).id
    skill = h.repo.add_skill(uid, name, summary, body.strip(), "authored")
    print(f"Added skill '{skill.name}' for user '{user_id}' (id={skill.id}).")
    return 0


def list_skills(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: harness list-skills <user_id>")
        return 1
    h = Harness(load_config())
    uid = h.repo.get_or_create_user(argv[2]).id
    skills = h.repo.list_skills(uid)
    if not skills:
        print(f"No skills for user '{argv[2]}'.")
        return 0
    for s in skills:
        print(f"- {s.name} [{s.origin}] — {s.summary}")
    return 0


def chat() -> int:
    cfg = load_config()
    h = Harness(cfg, echo=True)
    backend = "Postgres" if cfg.database_url else "in-memory"
    provider = "OpenRouter" if cfg.openrouter_api_key else "FakeProvider (offline)"
    print(f"Harness ready · repo={backend} · provider={provider} · model={cfg.model}")

    # connect remote HTTP MCP servers declared in the env (e.g. Fellow)
    for srv in mcp_http_servers():
        try:
            client = h.add_mcp_http(srv["url"], srv["name"], srv["headers"],
                                    oauth=srv.get("oauth"))
            print(f"MCP '{srv['name']}' connected · {len(client.list_tools())} tools "
                  f"indexed from {srv['url']}")
        except Exception as e:
            print(f"MCP '{srv['name']}' FAILED ({srv['url']}): {e}")

    external_id = input("user id: ").strip() or "demo-user"
    session = h.start_session(external_id)
    budget_label = session.token_budget or "unlimited"
    print(f"session {session.id} (context_window={session.context_window}, "
          f"budget={budget_label})")
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
    for client in h.tools.mcp_clients.values():
        try:
            client.stop()
        except Exception:
            pass
    print("session closed.")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "chat"
    if cmd == "init-db":
        return init_db()
    if cmd == "chat":
        return chat()
    if cmd == "add-skill":
        return add_skill(argv)
    if cmd == "list-skills":
        return list_skills(argv)
    print(__doc__)
    return 1


def entry() -> int:
    """Console-script entry point (`harness ...`)."""
    return main(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
