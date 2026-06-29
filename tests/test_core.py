"""Core unit tests — run with: python3 -m pytest -q  (or python3 tests/test_core.py)."""
from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.app import Harness
from harness.config import Config
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


def test_search_skills_keyword_and_get_skill():
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    uid = h.repo.get_or_create_user("u1").id
    h.repo.add_skill(uid, "deploy_web", "Ship the web app",
                     "1. run tests\n2. push to prod", "authored")
    h.repo.add_skill(uid, "rotate_keys", "Rotate API credentials",
                     "1. mint new key\n2. revoke old", "authored")
    s = h.start_session("u1")
    # SearchSkills returns name + summary only (progressive disclosure)
    found = h.tools.dispatch(s, {"id": "1", "function": {
        "name": "SearchSkills", "arguments": '{"query": "credentials"}'}})["content"]
    assert "rotate_keys" in found and "deploy_web" not in found
    assert "mint new key" not in found            # body is NOT in the summary view
    # GetSkill returns the full body by name
    body = h.tools.dispatch(s, {"id": "2", "function": {
        "name": "GetSkill", "arguments": '{"name": "rotate_keys"}'}})["content"]
    assert "mint new key" in body and "revoke old" in body
    # unknown name is handled gracefully
    miss = h.tools.dispatch(s, {"id": "3", "function": {
        "name": "GetSkill", "arguments": '{"name": "nope"}'}})["content"]
    assert "not found" in miss.lower()


def test_skills_injected_into_prompt_per_user():
    import datetime
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    alice = h.repo.get_or_create_user("alice").id
    h.repo.add_skill(alice, "deploy_web", "Ship the web app to prod",
                     "1. run tests\n2. push to prod", "authored")
    sa = h.start_session("alice")
    p.queue(content="ok")
    h.run_turn(sa, "hi")
    sys_msg = p.calls[-1][0]["content"]
    # catalog lists name + summary...
    assert "deploy_web" in sys_msg and "Ship the web app to prod" in sys_msg
    # ...but NOT the body (steps)
    assert "run tests" not in sys_msg
    # placed after the stable base prompt, before the volatile date (caching):
    # base ("sys.") < catalog heading < today's date
    assert sys_msg.index("sys.") < sys_msg.index("# Your saved skills")
    assert sys_msg.index("# Your saved skills") < sys_msg.index("Today's date is")
    assert sys_msg.rstrip().endswith(
        f"Today's date is {datetime.date.today().isoformat()}.")
    # a different tenant with no skills gets no catalog (isolation)
    sb = h.start_session("bob")
    p.queue(content="ok")
    h.run_turn(sb, "hi")
    assert "deploy_web" not in p.calls[-1][0]["content"]
    assert "# Your saved skills" not in p.calls[-1][0]["content"]


def test_skills_block_truncates_above_limit():
    from harness.prompt import skills_block
    from harness.models import Skill
    skills = [Skill(id=str(i), user_id="u", name=f"s{i}", summary="x", body="b")
              for i in range(35)]
    block = skills_block(skills, limit=30)
    assert "s0:" in block and "s29:" in block
    assert "s30:" not in block            # beyond the limit
    assert "5 more not shown" in block and "SearchSkills" in block


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


def test_soul_default_and_custom():
    from harness.prompt import build_system_prompt, load_soul, DEFAULT_IDENTITY
    # comment-only / empty -> default identity
    assert load_soul(explicit="<!-- just a comment -->") == DEFAULT_IDENTITY
    assert load_soul(explicit="   ") == DEFAULT_IDENTITY
    # explicit persona wins and tool guidance (incl. Bash policy) is appended
    sp = build_system_prompt(soul="You are Atlas, a terse SRE.")
    assert "Atlas" in sp
    assert "Bash" in sp and "fallback" in sp.lower()


def test_soul_loaded_from_file(tmp_path):
    from harness.prompt import build_system_prompt
    soul = tmp_path / "SOUL.md"
    soul.write_text("<!-- header -->\nYou are Nyx, a poetic assistant.")
    sp = build_system_prompt(str(soul))
    assert "Nyx" in sp and "poetic" in sp


def test_harness_uses_soul():
    import dataclasses
    from harness.app import Harness
    from harness.config import Config
    p = FakeProvider(context_window=4000)
    h = Harness(dataclasses.replace(Config(), database_url=""),
                soul="You are Atlas.", provider=p)
    assert "Atlas" in h.loop.system_prompt


def test_today_line_format():
    from datetime import date
    from harness.prompt import today_line, with_today
    assert today_line(date(2026, 6, 25)) == "Today's date is 2026-06-25."
    sp = with_today("BASE PROMPT", date(2026, 1, 2))
    assert sp.endswith("Today's date is 2026-01-02.")
    assert sp.startswith("BASE PROMPT")


def test_loop_appends_today_date_to_system_prompt():
    import datetime
    p = FakeProvider(context_window=4000)
    p.queue(content="ok")
    h = _harness(p)
    s = h.start_session("u1")
    h.run_turn(s, "hi")
    # the system message sent to the provider must carry today's date at the end
    sent = p.calls[-1]
    system_msg = sent[0]["content"]
    assert system_msg.rstrip().endswith(
        f"Today's date is {datetime.date.today().isoformat()}.")


def test_bash_cwd_persists_across_calls():
    import json
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    s = h.start_session("u1")

    def bash(cmd):
        return h.tools.dispatch(s, {"id": "x", "function": {
            "name": "Bash", "arguments": json.dumps({"command": cmd})}})["content"]

    bash("mkdir -p a/b && cd a/b")
    out = bash("pwd")
    assert out.count("a/b") >= 1 and "<exit_code>0</exit_code>" in out


def test_bash_reports_exit_and_stderr():
    import json
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    s = h.start_session("u1")
    out = h.tools.dispatch(s, {"id": "x", "function": {
        "name": "Bash", "arguments": json.dumps({"command": "ls /no/such/path"})}})["content"]
    assert "<exit_code>" in out and "<exit_code>0</exit_code>" not in out
    assert "<stderr>" in out


def test_bash_truncates_large_output():
    from harness.sandbox import LocalSubprocessSandbox
    sb = LocalSubprocessSandbox(max_output=500)
    res = sb.exec("sess", "for i in $(seq 1 2000); do echo line-$i; done")
    assert "characters elided" in res.stdout
    assert len(res.stdout) < 800
    sb.destroy("sess")


def test_unlimited_token_budget():
    p = FakeProvider(context_window=4000)
    # would-be expensive run: many tool calls, but budget 0 == no limit
    for _ in range(6):
        p.queue(content="...", tool_calls=[{"id": "x", "function": {
            "name": "Bash", "arguments": '{"command": "echo k"}'}}])
    p.queue(content="finished")
    h = _harness(p, token_budget_per_session=0)
    s = h.start_session("u1")
    r = h.run_turn(s, "do a lot")
    assert r.status == "ok" and r.text == "finished"   # never budget_exhausted
    assert h.repo.get_session(s.id).status != "budget_exhausted"


def test_budget_config_unlimited_parsing():
    import os
    from harness.config import _budget
    for val in ("0", "none", "unlimited", "-1"):
        os.environ["TOKEN_BUDGET_PER_SESSION"] = val
        try:
            assert _budget("TOKEN_BUDGET_PER_SESSION", 500_000) == 0, val
        finally:
            os.environ.pop("TOKEN_BUDGET_PER_SESSION", None)
    os.environ["TOKEN_BUDGET_PER_SESSION"] = "12345"
    try:
        assert _budget("TOKEN_BUDGET_PER_SESSION", 500_000) == 12345
    finally:
        os.environ.pop("TOKEN_BUDGET_PER_SESSION", None)


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


def test_mcp_http_servers_config():
    import os
    from harness.config import mcp_http_servers
    os.environ["MCP_HTTP_SERVERS"] = "fellow=https://fellow.app/mcp, other=https://x/mcp"
    os.environ["MCP_FELLOW_TOKEN"] = "tok123"
    os.environ.pop("MCP_OTHER_TOKEN", None)
    try:
        servers = mcp_http_servers()
        by_name = {s["name"]: s for s in servers}
        assert by_name["fellow"]["url"] == "https://fellow.app/mcp"
        assert by_name["fellow"]["headers"]["Authorization"] == "Bearer tok123"
        assert by_name["other"]["headers"] == {}   # no token -> no auth header
    finally:
        os.environ.pop("MCP_HTTP_SERVERS", None)
        os.environ.pop("MCP_FELLOW_TOKEN", None)


def test_oauth_pkce_and_helpers():
    import base64, hashlib
    from harness.oauth import make_pkce, origin, parse_resource_metadata_url, OAuthClient
    verifier, challenge = make_pkce()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected                       # S256 binding holds
    assert origin("https://fellow.app/mcp/x?y=1") == "https://fellow.app"
    assert parse_resource_metadata_url(
        'Bearer resource_metadata="https://as/.well-known/x"') == "https://as/.well-known/x"
    assert parse_resource_metadata_url("Bearer realm=foo") is None
    assert OAuthClient._expired({"obtained_at": 0, "expires_in": 3600}) is True
    assert OAuthClient._expired({}) is False           # no expiry info -> not expired


def test_oauth_token_cache(tmp_path):
    from harness.oauth import OAuthClient, OAuthConfig
    oc = OAuthClient(http_client=None, cfg=OAuthConfig(cache_dir=tmp_path))
    oc._save_cache("fellow.app", {"client_id": "c1", "tokens": {"access_token": "a"}})
    loaded = oc._load_cache("fellow.app")
    assert loaded["client_id"] == "c1" and loaded["tokens"]["access_token"] == "a"
    assert oc._load_cache("unknown.host") == {}


def test_http_mcp_oauth_retry_on_401():
    """A 401 triggers the OAuth flow (stubbed) and the request is retried with a token."""
    from harness.mcp_client import HttpMcpClient

    class Resp401:
        status_code = 401
        headers = {"www-authenticate": 'Bearer resource_metadata="https://as/meta"'}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b""
        def raise_for_status(self): raise AssertionError("should retry, not raise")

    class RespOK:
        status_code = 200
        headers = {"content-type": "application/json"}
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def read(self):
            import json
            return json.dumps(self._p)

    class StubHttp:
        def __init__(self): self.calls = 0
        def stream(self, method, url, json, headers):
            self.calls += 1
            if self.calls == 1:                         # first attempt: unauthorized
                assert "Authorization" not in headers
                return Resp401()
            assert headers.get("Authorization") == "Bearer tok-xyz"
            return RespOK({"jsonrpc": "2.0", "id": json["id"], "result": {"ok": True}})

    c = HttpMcpClient("https://srv/mcp", oauth=True)
    c._client = StubHttp()
    # stub the OAuth dance so no browser/network is needed
    c._authorize = lambda www: c._extra.__setitem__("Authorization", "Bearer tok-xyz")
    assert c._request("tools/list", {}) == {"ok": True}


def test_http_mcp_sse_parser():
    from harness.mcp_client import HttpMcpClient
    lines = [
        "event: message",
        'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18"}}',
        "",
        ": keep-alive comment",
        'data: {"jsonrpc":"2.0","id":2,"result":{"tools":[]}}',
    ]
    msg = HttpMcpClient._find_response_in_sse(iter(lines), 2)
    assert msg is not None and msg["id"] == 2 and msg["result"]["tools"] == []
    assert HttpMcpClient._find_response_in_sse(iter(lines), 99) is None


def test_http_mcp_call_via_stub_transport():
    """End-to-end dispatch of an HTTP MCP tool using a stubbed httpx client."""
    from harness.mcp_client import HttpMcpClient, ingest_server
    p = FakeProvider(context_window=4000)
    h = _harness(p)

    class StubResp:
        status_code = 200
        def __init__(self, payload):
            self._payload = payload
            self.headers = {"content-type": "application/json"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def read(self): import json; return json.dumps(self._payload)

    class StubHttp:
        def stream(self, method, url, json, headers):
            m = json["method"]
            if m == "initialize":
                return StubResp({"jsonrpc": "2.0", "id": json["id"],
                                 "result": {"protocolVersion": "2025-06-18"}})
            if m == "tools/list":
                return StubResp({"jsonrpc": "2.0", "id": json["id"], "result": {
                    "tools": [{"name": "get_meetings",
                               "description": "List recent meetings", "inputSchema": {}}]}})
            if m == "tools/call":
                return StubResp({"jsonrpc": "2.0", "id": json["id"],
                                 "result": {"meetings": ["standup"]}})
            return StubResp({"jsonrpc": "2.0", "id": json["id"], "result": {}})
        def post(self, url, json, headers): pass  # notifications
        def close(self): pass

    client = HttpMcpClient("https://example.test/mcp", name="stub")
    client._client = StubHttp()                       # inject stub transport
    client._notify("notifications/initialized", {})   # no-op
    ingest_server(h.repo, client)
    h.tools.mcp_clients["stub"] = client

    s = h.start_session("u1")
    # found via keyword search
    found = h.tools.dispatch(s, {"id": "1", "function": {
        "name": "SearchTools", "arguments": '{"query": "recent meetings"}'}})
    assert "get_meetings" in found["content"]
    # dispatched through the HTTP client
    out = h.tools.dispatch(s, {"id": "2", "function": {
        "name": "get_meetings", "arguments": "{}"}})
    assert "standup" in out["content"]


# --------------------------------------------------------------------------
# Streaming: provider.stream, loop.run_turn_stream, OpenRouter SSE accumulation
# --------------------------------------------------------------------------

def _drain(gen):
    """Consume a stream() generator: return (deltas, final ModelResult)."""
    deltas = []
    while True:
        try:
            deltas.append(next(gen))
        except StopIteration as stop:
            return deltas, stop.value


def test_fake_stream_matches_complete():
    """FakeProvider.stream deltas concatenate to the full content and its
    ModelResult tokens match complete() (so run_turn's TurnResult is unchanged)."""
    p = FakeProvider(context_window=4000)
    p.queue(content="hello streamed world")
    deltas, res = _drain(p.stream("m", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == "hello streamed world"
    assert len(deltas) >= 2                       # actually chunked, not one blob
    # token math identical to complete() on the same script/input
    p2 = FakeProvider(context_window=4000)
    p2.queue(content="hello streamed world")
    comp = p2.complete("m", [{"role": "user", "content": "hi"}])
    assert (res.tokens_in, res.tokens_out) == (comp.tokens_in, comp.tokens_out)


def test_run_turn_stream_event_order_and_final():
    p = FakeProvider(context_window=4000)
    p.queue(content="thinking", tool_calls=[{"id": "c1", "function": {
        "name": "Bash", "arguments": '{"command": "echo xyz"}'}}])
    p.queue(content="all done")
    h = _harness(p)
    s = h.start_session("u1")
    events = list(h.run_turn_stream(s, "go"))
    kinds = [e.kind for e in events]
    # text* then tool_start, tool_result, then text* then final
    assert kinds[0] == "text"
    assert "tool_start" in kinds and "tool_result" in kinds
    assert kinds.index("tool_start") < kinds.index("tool_result")
    assert kinds[-1] == "final"
    start = next(e for e in events if e.kind == "tool_start")
    assert start.name == "Bash" and start.args == {"command": "echo xyz"}
    assert start.call_id == "c1"
    result = next(e for e in events if e.kind == "tool_result")
    assert "xyz" in result.content
    final = events[-1].result
    assert final.status == "ok" and final.steps == 2


def test_run_turn_equals_streamed_final():
    """run_turn (the drainer) returns exactly the stream's final TurnResult."""
    def make():
        p = FakeProvider(context_window=4000)
        p.queue(content="x", tool_calls=[{"id": "c1", "function": {
            "name": "Bash", "arguments": '{"command": "echo hi"}'}}])
        p.queue(content="final answer")
        return _harness(p)
    h1 = make(); r1 = h1.run_turn(h1.start_session("u1"), "go")
    h2 = make()
    streamed = [e for e in h2.run_turn_stream(h2.start_session("u1"), "go")
                if e.kind == "final"][0].result
    assert dataclasses.astuple(r1) == dataclasses.astuple(streamed)


def test_openrouter_stream_sse_accumulation():
    """OpenRouter SSE: split content + fragmented tool_call arguments assemble
    into one ModelResult; usage from the final chunk is captured."""
    import json as _json

    from harness.provider import OpenRouterProvider

    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "Bash", "arguments": '{"comm'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'and": "ls"}'}}]}}]},
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 11,
                                               "completion_tokens": 7}},
    ]
    lines = [f"data: {_json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    class StubStreamResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self): return iter(lines)

    class StubClient:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stream(self, method, url, json):
            assert json["stream"] is True
            assert json["stream_options"] == {"include_usage": True}
            return StubStreamResp()

    prov = OpenRouterProvider(Config())
    prov._client = lambda: StubClient()
    deltas, res = _drain(prov.stream("m", [{"role": "user", "content": "hi"}],
                                     tools=[{"type": "function"}]))
    assert "".join(deltas) == "Hello"
    assert res.text == "Hello"
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'
    assert res.tokens_in == 11 and res.tokens_out == 7


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
