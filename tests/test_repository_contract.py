"""One battery, two backends: the Repository contract.

Every test here runs against InMemoryRepository (always) AND against
PostgresRepository (opt-in): the two implementations must be observably
identical, so the production backend is no longer the only untested one.

Run the Postgres side with:

    make db-up
    make test-db          # = TEST_DATABASE_URL=postgresql://harness:harness@localhost:5433/harness
                          #   uv run pytest -q -m integration

Without TEST_DATABASE_URL the postgres params skip themselves; the memory
params always run. The schema is (re-)applied from schema.sql, which is
idempotent — so this suite also detects drift between schema.sql and the SQL
in PostgresRepository.
"""

from __future__ import annotations

import os
import uuid

import pytest

from harness.persistence.repository import InMemoryRepository, PostgresRepository

# Order matters only for readability; TRUNCATE ... CASCADE handles FKs.
_TABLES = "step_logs, checkpoints, summaries, turns, skills, tool_index, sessions, users"


@pytest.fixture(scope="session")
def _pg():
    dsn = os.environ.get("TEST_DATABASE_URL", "")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set — Postgres contract tests are opt-in")
    repo = PostgresRepository(dsn)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "schema.sql")) as f:
        repo.conn.execute(f.read())
    yield repo
    repo.conn.close()


@pytest.fixture
def pg_repo(_pg):
    _pg.conn.execute(f"TRUNCATE {_TABLES} CASCADE")
    return _pg


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
def repo(request):
    if request.param == "memory":
        return InMemoryRepository()
    return request.getfixturevalue("pg_repo")


def _user_session(repo, external_id="u1", **kw):
    user = repo.get_or_create_user(external_id)
    session = repo.create_session(
        user.id, kw.get("model", "m"), kw.get("context_window", 4000), kw.get("token_budget", 0)
    )
    return user, session


# --------------------------------------------------------------------------
# users & sessions
# --------------------------------------------------------------------------


def test_get_or_create_user_is_idempotent(repo):
    a1 = repo.get_or_create_user("alice")
    a2 = repo.get_or_create_user("alice")
    b = repo.get_or_create_user("bob")
    assert a1.id == a2.id and a1.external_id == "alice"
    assert b.id != a1.id


def test_session_roundtrip_get_and_find(repo):
    user, s = _user_session(repo, model="gpt-x", context_window=1234, token_budget=99)
    got = repo.get_session(s.id)
    assert (got.id, got.user_id) == (s.id, user.id)
    assert (got.model, got.context_window, got.token_budget) == ("gpt-x", 1234, 99)
    assert got.tokens_spent == 0 and got.status == "open"
    assert repo.find_session(s.id).id == s.id


def test_find_session_returns_none_for_unknown_and_malformed_ids(repo):
    # unknown-but-well-formed id
    assert repo.find_session(str(uuid.uuid4())) is None
    # malformed id (not a uuid): the contract says None, not an exception —
    # this is exactly what Harness.start_session(session_id=...) relies on
    assert repo.find_session("does-not-exist") is None


def test_session_tokens_status_close_and_count(repo):
    user, s = _user_session(repo)
    repo.add_session_tokens(s.id, 10)
    repo.add_session_tokens(s.id, 5)
    assert repo.get_session(s.id).tokens_spent == 15

    repo.set_session_status(s.id, "budget_exhausted")
    assert repo.get_session(s.id).status == "budget_exhausted"

    assert repo.count_closed_sessions(user.id) == 0
    repo.close_session(s.id)
    assert repo.get_session(s.id).status == "closed"
    assert repo.count_closed_sessions(user.id) == 1


# --------------------------------------------------------------------------
# turns & the active window
# --------------------------------------------------------------------------


def test_turn_idx_sequence_and_content_roundtrip(repo):
    user, s = _user_session(repo)
    assistant_msg = {
        "role": "assistant",
        "content": "sure",
        "tool_calls": [{"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}],
    }
    t0 = repo.add_turn(s.id, user.id, "user", "hello", 3)
    t1 = repo.add_turn(s.id, user.id, "assistant", assistant_msg, 7, tokens_in=11, tokens_out=5)
    t2 = repo.add_turn(s.id, user.id, "tool", {"tool_call_id": "c1", "content": "file.txt"}, 2)
    assert (t0.idx, t1.idx, t2.idx) == (0, 1, 2)

    active = repo.active_turns(s.id)
    assert [t.idx for t in active] == [0, 1, 2]
    assert [t.role for t in active] == ["user", "assistant", "tool"]
    # content survives the backend roundtrip with its Python shape intact
    assert active[0].content == "hello"
    assert active[1].content == assistant_msg
    assert active[2].content == {"tool_call_id": "c1", "content": "file.txt"}
    assert active[1].tokens_in == 11 and active[1].tokens_out == 5
    # a second session starts its own idx sequence
    _, s2 = _user_session(repo, "u2")
    assert repo.add_turn(s2.id, repo.get_or_create_user("u2").id, "user", "hi", 1).idx == 0


def test_mark_out_of_window_folds_turns_and_empty_is_a_noop(repo):
    user, s = _user_session(repo)
    t0 = repo.add_turn(s.id, user.id, "user", "old", 1)
    t1 = repo.add_turn(s.id, user.id, "assistant", "new", 1)
    repo.mark_out_of_window([t0.id])
    assert [t.id for t in repo.active_turns(s.id)] == [t1.id]
    repo.mark_out_of_window([])  # must not raise
    assert [t.id for t in repo.active_turns(s.id)] == [t1.id]


def test_count_user_turns_and_user_turns_since(repo):
    user, s = _user_session(repo)
    repo.add_turn(s.id, user.id, "user", "q1", 1)
    repo.add_turn(s.id, user.id, "assistant", "a1", 1)
    repo.add_turn(s.id, user.id, "user", "q2", 1)
    repo.add_turn(s.id, user.id, "tool", {"tool_call_id": "x", "content": "out"}, 1)
    repo.add_turn(s.id, user.id, "user", "q3", 1)

    assert repo.count_user_turns(s.id) == 3
    since = repo.user_turns_since(s.id, 1)  # after the 1st user turn
    assert [t.content for t in since] == ["q2", "q3"]
    assert all(t.role == "user" for t in since)


# --------------------------------------------------------------------------
# summaries & checkpoints
# --------------------------------------------------------------------------


def test_summary_chain_and_current(repo):
    _, s = _user_session(repo)
    assert repo.current_summary(s.id) is None

    s1 = repo.add_summary(s.id, parent_id=None, content="first fold", token_count=3, covers_until=4)
    assert repo.current_summary(s.id).id == s1.id
    s2 = repo.add_summary(s.id, parent_id=s1.id, content="second fold", token_count=3, covers_until=9)

    cur = repo.current_summary(s.id)
    assert cur.id == s2.id and cur.parent_id == s1.id
    assert cur.content == "second fold" and cur.covers_until == 9


def test_checkpoints_and_last_checkpoint_turn(repo):
    user, s = _user_session(repo)
    assert repo.last_checkpoint_turn(s.id) == 0  # none yet
    repo.add_checkpoint(s.id, user.id, 3, "postgres tuning")
    repo.add_checkpoint(s.id, user.id, 6, "index design")
    assert repo.last_checkpoint_turn(s.id) == 6


# --------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------


def test_skills_add_list_get_and_keyword_search(repo):
    user, _ = _user_session(repo)
    repo.add_skill(user.id, "deploy_web", "Ship the web app", "1. test\n2. deploy", "authored")
    repo.add_skill(user.id, "rotate_keys", "Rotate API credentials", "1. mint\n2. revoke", "induced")

    names = {sk.name for sk in repo.list_skills(user.id)}
    assert names == {"deploy_web", "rotate_keys"}

    got = repo.get_skill(user.id, "rotate_keys")
    assert got is not None and got.origin == "induced" and "mint" in got.body
    assert repo.get_skill(user.id, "nope") is None

    found = repo.search_skills(user.id, "credentials", k=5)
    assert [sk.name for sk in found] == ["rotate_keys"]

    # tenant isolation: another user sees nothing
    other = repo.get_or_create_user("someone-else")
    assert repo.list_skills(other.id) == []
    assert repo.search_skills(other.id, "credentials", k=5) == []


# --------------------------------------------------------------------------
# tool index
# --------------------------------------------------------------------------


def test_tool_upsert_updates_in_place_and_get(repo):
    repo.upsert_tool("mail", "send_email", "Send an email message", {"type": "object"})
    repo.upsert_tool("mail", "send_email", "Send an email (v2)", {"type": "object", "x": 1})
    got = repo.get_tool("send_email")
    assert got is not None
    assert got.description == "Send an email (v2)"
    assert got.input_schema == {"type": "object", "x": 1}
    assert repo.get_tool("missing_tool") is None
    # the upsert updated, not duplicated
    assert len(repo.search_tools("send_email", k=10)) == 1


def test_search_tools_keyword_and_accent_insensitive(repo):
    repo.upsert_tool("mail", "send_email", "Send an email message", {})
    repo.upsert_tool("fs", "read_file", "Read a file from disk", {})
    repo.upsert_tool("cal", "agendar_reuniao", "Agendar uma reunião com o cliente", {})

    names = [t.name for t in repo.search_tools("email the customer", k=5)]
    assert "send_email" in names and "read_file" not in names

    # accent-insensitive in BOTH directions: bare and accented queries hit
    # the accented description ("multilingual by design")
    for query in ("reuniao", "reunião"):
        names = [t.name for t in repo.search_tools(query, k=5)]
        assert names == ["agendar_reuniao"], f"query {query!r} missed"


# --------------------------------------------------------------------------
# session listing & step logs
# --------------------------------------------------------------------------


def test_list_sessions_ordering_subject_and_counts(repo):
    user = repo.get_or_create_user("u1")
    s1 = repo.create_session(user.id, "m", 1000, 0)
    repo.add_turn(s1.id, user.id, "user", "first thing", 3)
    s2 = repo.create_session(user.id, "m", 1000, 0)
    repo.add_turn(s2.id, user.id, "user", "second thing", 3)
    repo.add_turn(s2.id, user.id, "assistant", "reply", 3)

    rows = repo.list_sessions(user.id)
    assert [r.id for r in rows] == [s2.id, s1.id]  # newest first
    assert rows[0].subject == "second thing" and rows[0].turns == 2
    assert rows[1].subject == "first thing" and rows[1].turns == 1

    # a checkpoint label takes precedence over the first-message snippet
    repo.add_checkpoint(s1.id, user.id, 1, "boot sequence")
    rows = repo.list_sessions(user.id)
    assert next(r for r in rows if r.id == s1.id).subject == "boot sequence"

    # another user's picker is empty (isolation)
    assert repo.list_sessions(repo.get_or_create_user("u2").id) == []


def test_add_step_log_accepts_full_and_minimal_rows(repo):
    user, s = _user_session(repo)
    t = repo.add_turn(s.id, user.id, "user", "hi", 1)
    # full row
    repo.add_step_log(s.id, t.id, "model_call", {"step": 0}, tokens_in=5, tokens_out=2, latency_ms=17)
    # minimal row: no session/turn (e.g. app-level events) and empty detail
    repo.add_step_log(None, None, "hook_error", {})
