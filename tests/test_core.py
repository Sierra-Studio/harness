"""Core unit tests — run with: python3 -m pytest -q  (or python3 tests/test_core.py)."""
from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.app import Harness
from harness.config import Config
from harness.embeddings import Embedder, cosine
from harness.provider import FakeProvider
from harness.tokenizer import count_tokens


def _harness(provider, **overrides):
    # force in-memory repo so tests never depend on ambient .env / DATABASE_URL
    cfg = dataclasses.replace(Config(), database_url="", **overrides)
    return Harness(cfg, system_prompt="sys.", provider=provider)


def test_tokenizer_monotonic():
    # empty content is 0 with a real tokenizer, >=1 with the heuristic fallback
    assert count_tokens("") >= 0
    assert count_tokens("a" * 400) > count_tokens("a" * 4)
    assert count_tokens({"role": "user", "content": "hi"}) >= 1


def test_embeddings_self_similarity():
    emb = Embedder(Config())
    a = emb.embed("send an email to the customer")
    b = emb.embed("send an email to the customer")
    c = emb.embed("compile a rust binary")
    assert cosine(a, b) > cosine(a, c)


def test_loop_basic_reply():
    p = FakeProvider(context_window=4000)
    p.queue(content="hello there")
    h = _harness(p)
    s = h.start_session("u1")
    r = h.run_turn(s, "hi")
    assert r.status == "ok" and r.text == "hello there"
    assert r.tokens_spent > 0


def test_loop_tool_call_runs_bash():
    p = FakeProvider(context_window=4000)
    p.queue(tool_calls=[{"id": "c1", "function": {
        "name": "Bash", "arguments": '{"command": "echo xyz"}'}}])
    p.queue(content="done")
    h = _harness(p)
    s = h.start_session("u1")
    r = h.run_turn(s, "run it")
    assert r.status == "ok" and r.steps == 2
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and "xyz" in tool_turns[0].content["content"]


def test_budget_guard_returns_partial():
    p = FakeProvider(context_window=4000)
    for _ in range(10):
        p.queue(content="...", tool_calls=[{"id": "x", "function": {
            "name": "Bash", "arguments": '{"command": "echo k"}'}}])
    h = _harness(p, token_budget_per_session=20)
    s = h.start_session("u1")
    r = h.run_turn(s, "expensive")
    assert r.status == "budget_exhausted"
    assert h.repo.get_session(s.id).status == "budget_exhausted"


def test_summarization_compresses_window():
    p = FakeProvider(context_window=140)
    h = _harness(p, response_reserve_tokens=10, summary_keep_ratio=0.2)
    s = h.start_session("u1")
    for i in range(6):
        p.queue(content="answer " + "z" * 80)
        h.run_turn(s, "question " + "w" * 60)
    assert len(h.repo.summaries) >= 1
    active = h.repo.active_turns(s.id)
    total = [t for t in h.repo.turns if t.session_id == s.id]
    assert len(active) < len(total)          # window is a compressed projection
    # chaining: latest summary points back to a parent after >1 fold
    if len(h.repo.summaries) > 1:
        assert h.repo.summaries[-1].parent_id is not None


def test_checkpoint_every_n_user_turns():
    p = FakeProvider(context_window=4000)
    h = _harness(p, checkpoint_every_user_turns=3)
    s = h.start_session("u1")
    for i in range(3):
        p.queue(content="ok")
        h.run_turn(s, f"q{i}")
    assert len(h.repo.checkpoints) == 1
    assert h.repo.checkpoints[0]["at_user_turn"] == 3


def test_multi_tenant_isolation():
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    sa = h.start_session("alice")
    p.queue(content="a"); h.run_turn(sa, "hi from alice")
    sb = h.start_session("bob")
    p.queue(content="b"); h.run_turn(sb, "hi from bob")
    bob_id = h.repo.get_or_create_user("bob").id
    bob_turns = [t for t in h.repo.turns if t.user_id == bob_id]
    assert bob_turns and all(t.user_id == bob_id for t in bob_turns)
    assert sa.id != sb.id


def test_skill_induction_and_dedup():
    class P(FakeProvider):
        def induce_skills(self, model, signals):
            return [{"name": "n", "summary": "s", "body": "do x"}]
    p = P(context_window=4000)
    h = _harness(p, skill_induction_every_sessions=2)
    uid = h.repo.get_or_create_user("alice").id
    for _ in range(2):
        s = h.start_session("alice")
        h.close_session(s)
    skills = h.repo.list_skills(uid)
    assert len(skills) == 1 and skills[0].origin == "induced"
    # another cadence hit must NOT create a duplicate
    for _ in range(2):
        s = h.start_session("alice")
        h.close_session(s)
    assert len(h.repo.list_skills(uid)) == 1


def test_search_tools_keyword_match():
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    h.repo.upsert_tool("mail", "send_email", "Send an email message", {})
    h.repo.upsert_tool("fs", "read_file", "Read a file from disk", {})
    s = h.start_session("u1")
    out = h.tools.dispatch(s, {"id": "1", "function": {
        "name": "SearchTools", "arguments": '{"query": "email the customer"}'}})
    assert "send_email" in out["content"]
    assert "read_file" not in out["content"]   # keyword search excludes non-matches


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
