"""Core unit tests — run with: python3 -m pytest -q."""

from __future__ import annotations

import dataclasses

from helpers import drain as _drain
from helpers import make_harness as _harness

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.llm.tokenizer import count_tokens
from harness.settings import Config, LoopConfig, MemoryConfig


def test_tokenizer_monotonic():
    # empty content is 0 with a real tokenizer, >=1 with the heuristic fallback
    assert count_tokens("") >= 0
    assert count_tokens("a" * 400) > count_tokens("a" * 4)
    assert count_tokens({"role": "user", "content": "hi"}) >= 1


def test_search_skills_keyword_and_get_skill():
    p = FakeProvider(context_window=4000)
    h = _harness(p)
    uid = h.repo.get_or_create_user("u1").id
    h.repo.add_skill(
        uid, "deploy_web", "Ship the web app", "1. run tests\n2. push to prod", "authored"
    )
    h.repo.add_skill(
        uid, "rotate_keys", "Rotate API credentials", "1. mint new key\n2. revoke old", "authored"
    )
    s = h.start_session("u1")
    # SearchSkills returns name + summary only (progressive disclosure)
    found = h.tools.dispatch(
        s,
        {"id": "1", "function": {"name": "SearchSkills", "arguments": '{"query": "credentials"}'}},
    )["content"]
    assert "rotate_keys" in found and "deploy_web" not in found
    assert "mint new key" not in found  # body is NOT in the summary view
    # GetSkill returns the full body by name
    body = h.tools.dispatch(
        s, {"id": "2", "function": {"name": "GetSkill", "arguments": '{"name": "rotate_keys"}'}}
    )["content"]
    assert "mint new key" in body and "revoke old" in body
    # unknown name is handled gracefully
    miss = h.tools.dispatch(
        s, {"id": "3", "function": {"name": "GetSkill", "arguments": '{"name": "nope"}'}}
    )["content"]
    assert "not found" in miss.lower()


def test_skills_injected_into_prompt_per_user():
    import datetime

    p = FakeProvider(context_window=4000)
    h = _harness(p)
    alice = h.repo.get_or_create_user("alice").id
    h.repo.add_skill(
        alice, "deploy_web", "Ship the web app to prod", "1. run tests\n2. push to prod", "authored"
    )
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
    assert sys_msg.rstrip().endswith(f"Today's date is {datetime.date.today().isoformat()}.")
    # a different tenant with no skills gets no catalog (isolation)
    sb = h.start_session("bob")
    p.queue(content="ok")
    h.run_turn(sb, "hi")
    assert "deploy_web" not in p.calls[-1][0]["content"]
    assert "# Your saved skills" not in p.calls[-1][0]["content"]


def test_skills_block_truncates_above_limit():
    from harness.memory.persona import skills_block
    from harness.models import Skill

    skills = [Skill(id=str(i), user_id="u", name=f"s{i}", summary="x", body="b") for i in range(35)]
    block = skills_block(skills, limit=30)
    assert "s0:" in block and "s29:" in block
    assert "s30:" not in block  # beyond the limit
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
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo xyz"}'}}
        ]
    )
    p.queue(content="done")
    h = _harness(p)
    s = h.start_session("u1")
    r = h.run_turn(s, "run it")
    assert r.status == "ok" and r.steps == 2
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and "xyz" in tool_turns[0].content["content"]


def _queue_bash(p, cmd="echo xyz"):
    p.queue(tool_calls=[{"id": "c1", "function": {"name": "Bash", "arguments": f'{{"command": "{cmd}"}}'}}])
    p.queue(content="done")


def test_manual_mode_denies_tool_call():
    from harness.settings import PermissionConfig

    p = FakeProvider(context_window=4000)
    _queue_bash(p, "echo shouldnotrun")
    h = _harness(p, permissions=PermissionConfig(mode="manual"))
    from harness.core import DENY

    h.permissions.asker = lambda name, args: DENY
    s = h.start_session("u1")
    r = h.run_turn(s, "run it")
    assert r.status == "ok"
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and "denied by the user" in tool_turns[0].content["content"]
    assert "shouldnotrun" not in tool_turns[0].content["content"]


def test_manual_mode_always_remembers_for_session():
    from harness.core import ALWAYS
    from harness.settings import PermissionConfig

    p = FakeProvider(context_window=4000)
    _queue_bash(p, "echo first")
    _queue_bash(p, "echo second")  # a second turn's worth
    h = _harness(p, permissions=PermissionConfig(mode="manual"))
    asked = []
    h.permissions.asker = lambda name, args: (asked.append(name), ALWAYS)[1]
    s = h.start_session("u1")
    h.run_turn(s, "one")
    h.run_turn(s, "two")
    assert asked == ["Bash"]  # asked once; ALWAYS remembered for the rest of the session
    out = [t.content["content"] for t in h.repo.turns if t.role == "tool"]
    assert any("first" in c for c in out) and any("second" in c for c in out)


def test_auto_mode_never_asks():
    p = FakeProvider(context_window=4000)
    _queue_bash(p, "echo ran")
    h = _harness(p)  # default auto
    called = []
    h.permissions.asker = lambda name, args: called.append(name) or "deny"
    s = h.start_session("u1")
    h.run_turn(s, "go")
    assert called == []  # auto mode bypasses the asker entirely
    assert any("ran" in t.content["content"] for t in h.repo.turns if t.role == "tool")


def test_budget_guard_returns_partial():
    p = FakeProvider(context_window=4000)
    for _ in range(10):
        p.queue(
            content="...",
            tool_calls=[
                {"id": "x", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}
            ],
        )
    h = _harness(p, loop=LoopConfig(token_budget_per_session=20))
    s = h.start_session("u1")
    r = h.run_turn(s, "expensive")
    assert r.status == "budget_exhausted"
    assert h.repo.get_session(s.id).status == "budget_exhausted"


def test_persona_default_and_custom():
    from harness.memory.persona import DEFAULT_IDENTITY, build_system_prompt, load_persona
    from harness.tools.builtin import default_tools

    # comment-only / empty -> default identity
    assert load_persona(explicit="<!-- just a comment -->") == DEFAULT_IDENTITY
    assert load_persona(explicit="   ") == DEFAULT_IDENTITY
    # explicit persona wins and guidance composed from the active tools is appended
    sp = build_system_prompt(persona="You are Atlas, a terse SRE.", tools=default_tools())
    assert "Atlas" in sp
    assert "Bash" in sp and "fallback" in sp.lower()
    # without tools, no guidance layer is composed (persona only)
    assert "Bash" not in build_system_prompt(persona="You are Atlas, a terse SRE.")


def test_persona_loaded_from_file(tmp_path):
    from harness.memory.persona import build_system_prompt

    persona = tmp_path / "PERSONA.md"
    persona.write_text("<!-- header -->\nYou are Nyx, a poetic assistant.")
    sp = build_system_prompt(str(persona))
    assert "Nyx" in sp and "poetic" in sp


def test_harness_uses_persona():
    import dataclasses

    from harness.core import Harness
    from harness.settings import Config

    p = FakeProvider(context_window=4000)
    h = Harness(
        dataclasses.replace(Config(), database_url=""), persona="You are Atlas.", provider=p
    )
    assert "Atlas" in h.loop.system_prompt


def test_today_line_format():
    from datetime import date

    from harness.memory.persona import today_line, with_today

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
    assert system_msg.rstrip().endswith(f"Today's date is {datetime.date.today().isoformat()}.")


def test_bash_cwd_persists_across_calls():
    import json

    p = FakeProvider(context_window=4000)
    h = _harness(p)
    s = h.start_session("u1")

    def bash(cmd):
        return h.tools.dispatch(
            s, {"id": "x", "function": {"name": "Bash", "arguments": json.dumps({"command": cmd})}}
        )["content"]

    bash("mkdir -p a/b && cd a/b")
    out = bash("pwd")
    assert out.count("a/b") >= 1 and "<exit_code>0</exit_code>" in out


def test_bash_reports_exit_and_stderr():
    import json

    p = FakeProvider(context_window=4000)
    h = _harness(p)
    s = h.start_session("u1")
    out = h.tools.dispatch(
        s,
        {
            "id": "x",
            "function": {"name": "Bash", "arguments": json.dumps({"command": "ls /no/such/path"})},
        },
    )["content"]
    assert "<exit_code>" in out and "<exit_code>0</exit_code>" not in out
    assert "<stderr>" in out


def test_bash_truncates_large_output():
    from harness.tools.sandbox import LocalSubprocessSandbox

    sb = LocalSubprocessSandbox(max_output=500)
    res = sb.exec("sess", "for i in $(seq 1 2000); do echo line-$i; done")
    assert "characters elided" in res.stdout
    assert len(res.stdout) < 800
    sb.destroy("sess")


def _call(h, s, name, **args):
    import json

    return h.tools.dispatch(
        s, {"id": "x", "function": {"name": name, "arguments": json.dumps(args)}}
    )["content"]


def test_write_creates_file_then_edit_replaces():
    h = _harness(FakeProvider(context_window=4000))
    s = h.start_session("u1")
    out = _call(h, s, "Write", path="pkg/mod.py", content="a = 1\nb = 2\n")
    assert "Wrote 2 line" in out
    assert h.sandbox.read_file(s.id, "pkg/mod.py") == "a = 1\nb = 2\n"
    out = _call(h, s, "Edit", path="pkg/mod.py", old_string="a = 1", new_string="a = 99")
    assert "1 replacement" in out
    assert h.sandbox.read_file(s.id, "pkg/mod.py") == "a = 99\nb = 2\n"


def test_edit_requires_unique_match_unless_replace_all():
    h = _harness(FakeProvider(context_window=4000))
    s = h.start_session("u1")
    _call(h, s, "Write", path="d.txt", content="x\nx\n")
    dup = _call(h, s, "Edit", path="d.txt", old_string="x", new_string="y")
    assert "ERROR" in dup and "2 times" in dup
    ok = _call(h, s, "Edit", path="d.txt", old_string="x", new_string="y", replace_all=True)
    assert "2 replacements" in ok
    assert h.sandbox.read_file(s.id, "d.txt") == "y\ny\n"


def test_edit_errors_on_missing_file_and_absent_string():
    h = _harness(FakeProvider(context_window=4000))
    s = h.start_session("u1")
    assert "not found" in _call(h, s, "Edit", path="ghost", old_string="a", new_string="b")
    _call(h, s, "Write", path="f.txt", content="hello")
    assert "not found" in _call(h, s, "Edit", path="f.txt", old_string="NOPE", new_string="b")


def test_write_and_edit_share_bash_working_dir():
    h = _harness(FakeProvider(context_window=4000))
    s = h.start_session("u1")
    _call(h, s, "Bash", command="mkdir sub && cd sub")
    _call(h, s, "Write", path="in_sub.txt", content="here")
    # relative path resolved against the cwd Bash left us in (now inside sub/)
    cat = _call(h, s, "Bash", command="cat in_sub.txt")
    assert "here" in cat and "<exit_code>0</exit_code>" in cat


def test_unlimited_token_budget():
    p = FakeProvider(context_window=4000)
    # would-be expensive run: many tool calls, but budget 0 == no limit
    for _ in range(6):
        p.queue(
            content="...",
            tool_calls=[
                {"id": "x", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}
            ],
        )
    p.queue(content="finished")
    h = _harness(p, loop=LoopConfig(token_budget_per_session=0))
    s = h.start_session("u1")
    r = h.run_turn(s, "do a lot")
    assert r.status == "ok" and r.text == "finished"  # never budget_exhausted
    assert h.repo.get_session(s.id).status != "budget_exhausted"


def test_budget_config_unlimited_parsing():
    from harness.settings import _budget

    for val in ("0", "none", "unlimited", "-1"):
        env = {"TOKEN_BUDGET_PER_SESSION": val}
        assert _budget(env, "TOKEN_BUDGET_PER_SESSION", 500_000) == 0, val
    env = {"TOKEN_BUDGET_PER_SESSION": "12345"}
    assert _budget(env, "TOKEN_BUDGET_PER_SESSION", 500_000) == 12345


def test_summarization_compresses_window():
    p = FakeProvider(context_window=140)
    h = _harness(
        p,
        loop=LoopConfig(response_reserve_tokens=10),
        memory=MemoryConfig(summary_keep_ratio=0.2),
    )
    s = h.start_session("u1")
    for _ in range(6):
        p.queue(content="answer " + "z" * 80)
        h.run_turn(s, "question " + "w" * 60)
    assert len(h.repo.summaries) >= 1
    active = h.repo.active_turns(s.id)
    total = [t for t in h.repo.turns if t.session_id == s.id]
    assert len(active) < len(total)  # window is a compressed projection
    # chaining: latest summary points back to a parent after >1 fold
    if len(h.repo.summaries) > 1:
        assert h.repo.summaries[-1].parent_id is not None


def test_checkpoint_every_n_user_turns():
    p = FakeProvider(context_window=4000)
    h = _harness(p, memory=MemoryConfig(checkpoint_every_user_turns=3))
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
    p.queue(content="a")
    h.run_turn(sa, "hi from alice")
    sb = h.start_session("bob")
    p.queue(content="b")
    h.run_turn(sb, "hi from bob")
    bob_id = h.repo.get_or_create_user("bob").id
    bob_turns = [t for t in h.repo.turns if t.user_id == bob_id]
    assert bob_turns and all(t.user_id == bob_id for t in bob_turns)
    assert sa.id != sb.id


def test_skill_induction_and_dedup():
    class P(FakeProvider):
        def induce_skills(self, model, signals):
            return [{"name": "n", "summary": "s", "body": "do x"}]

    p = P(context_window=4000)
    h = _harness(p, memory=MemoryConfig(skill_induction_every_sessions=2))
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
    out = h.tools.dispatch(
        s,
        {
            "id": "1",
            "function": {"name": "SearchTools", "arguments": '{"query": "email the customer"}'},
        },
    )
    assert "send_email" in out["content"]
    assert "read_file" not in out["content"]  # keyword search excludes non-matches


def test_mcp_http_servers_config(monkeypatch):
    from harness.settings import mcp_http_servers

    monkeypatch.setenv("MCP_HTTP_SERVERS", "fellow=https://fellow.app/mcp, other=https://x/mcp")
    monkeypatch.setenv("MCP_FELLOW_TOKEN", "tok123")
    monkeypatch.delenv("MCP_OTHER_TOKEN", raising=False)
    servers = mcp_http_servers()
    by_name = {s["name"]: s for s in servers}
    assert by_name["fellow"]["url"] == "https://fellow.app/mcp"
    assert by_name["fellow"]["headers"]["Authorization"] == "Bearer tok123"
    assert by_name["other"]["headers"] == {}  # no token -> no auth header


def test_oauth_pkce_and_helpers():
    import base64
    import hashlib

    from harness.mcp.oauth import OAuthClient, make_pkce, origin, parse_resource_metadata_url

    verifier, challenge = make_pkce()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert challenge == expected  # S256 binding holds
    assert origin("https://fellow.app/mcp/x?y=1") == "https://fellow.app"
    assert (
        parse_resource_metadata_url('Bearer resource_metadata="https://as/.well-known/x"')
        == "https://as/.well-known/x"
    )
    assert parse_resource_metadata_url("Bearer realm=foo") is None
    assert OAuthClient._expired({"obtained_at": 0, "expires_in": 3600}) is True
    assert OAuthClient._expired({}) is False  # no expiry info -> not expired


def test_oauth_token_cache(tmp_path):
    from harness.mcp.oauth import OAuthClient, OAuthConfig

    oc = OAuthClient(http_client=None, cfg=OAuthConfig(cache_dir=tmp_path))
    oc._save_cache("fellow.app", {"client_id": "c1", "tokens": {"access_token": "a"}})
    loaded = oc._load_cache("fellow.app")
    assert loaded["client_id"] == "c1" and loaded["tokens"]["access_token"] == "a"
    assert oc._load_cache("unknown.host") == {}


def test_http_mcp_oauth_retry_on_401():
    """A 401 triggers the OAuth flow (stubbed) and the request is retried with a token."""
    from harness.mcp.client import HttpMcpClient

    class Resp401:
        status_code = 401
        headers = {"www-authenticate": 'Bearer resource_metadata="https://as/meta"'}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

        def raise_for_status(self):
            raise AssertionError("should retry, not raise")

    class RespOK:
        status_code = 200
        headers = {"content-type": "application/json"}

        def __init__(self, payload):
            self._p = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def read(self):
            import json

            return json.dumps(self._p)

    class StubHttp:
        def __init__(self):
            self.calls = 0

        def stream(self, method, url, json, headers):
            self.calls += 1
            if self.calls == 1:  # first attempt: unauthorized
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
    from harness.mcp.client import HttpMcpClient

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
    from harness.mcp.client import HttpMcpClient, ingest_server

    p = FakeProvider(context_window=4000)
    h = _harness(p)

    class StubResp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def read(self):
            import json

            return json.dumps(self._payload)

    class StubHttp:
        def stream(self, method, url, json, headers):
            m = json["method"]
            if m == "initialize":
                return StubResp(
                    {
                        "jsonrpc": "2.0",
                        "id": json["id"],
                        "result": {"protocolVersion": "2025-06-18"},
                    }
                )
            if m == "tools/list":
                return StubResp(
                    {
                        "jsonrpc": "2.0",
                        "id": json["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "get_meetings",
                                    "description": "List recent meetings",
                                    "inputSchema": {},
                                }
                            ]
                        },
                    }
                )
            if m == "tools/call":
                return StubResp(
                    {"jsonrpc": "2.0", "id": json["id"], "result": {"meetings": ["standup"]}}
                )
            return StubResp({"jsonrpc": "2.0", "id": json["id"], "result": {}})

        def post(self, url, json, headers):
            pass  # notifications

        def close(self):
            pass

    client = HttpMcpClient("https://example.test/mcp", name="stub")
    client._client = StubHttp()  # inject stub transport
    client._notify("notifications/initialized", {})  # no-op
    ingest_server(h.repo, client)
    h.tools.mcp_clients["stub"] = client

    s = h.start_session("u1")
    # found via keyword search
    found = h.tools.dispatch(
        s,
        {
            "id": "1",
            "function": {"name": "SearchTools", "arguments": '{"query": "recent meetings"}'},
        },
    )
    assert "get_meetings" in found["content"]
    # dispatched through the HTTP client
    out = h.tools.dispatch(s, {"id": "2", "function": {"name": "get_meetings", "arguments": "{}"}})
    assert "standup" in out["content"]


# --------------------------------------------------------------------------
# Streaming: provider.stream, loop.run_turn_stream, OpenRouter SSE accumulation
# --------------------------------------------------------------------------


def test_fake_stream_matches_complete():
    """FakeProvider.stream deltas concatenate to the full content and its
    ModelResult tokens match complete() (so run_turn's TurnResult is unchanged)."""
    p = FakeProvider(context_window=4000)
    p.queue(content="hello streamed world")
    deltas, res = _drain(p.stream("m", [{"role": "user", "content": "hi"}]))
    assert "".join(deltas) == "hello streamed world"
    assert len(deltas) >= 2  # actually chunked, not one blob
    # token math identical to complete() on the same script/input
    p2 = FakeProvider(context_window=4000)
    p2.queue(content="hello streamed world")
    comp = p2.complete("m", [{"role": "user", "content": "hi"}])
    assert (res.tokens_in, res.tokens_out) == (comp.tokens_in, comp.tokens_out)


def test_run_turn_stream_event_order_and_final():
    p = FakeProvider(context_window=4000)
    p.queue(
        content="thinking",
        tool_calls=[
            {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo xyz"}'}}
        ],
    )
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
        p.queue(
            content="x",
            tool_calls=[
                {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo hi"}'}}
            ],
        )
        p.queue(content="final answer")
        return _harness(p)

    h1 = make()
    r1 = h1.run_turn(h1.start_session("u1"), "go")
    h2 = make()
    streamed = [e for e in h2.run_turn_stream(h2.start_session("u1"), "go") if e.kind == "final"][
        0
    ].result
    assert dataclasses.astuple(r1) == dataclasses.astuple(streamed)


def test_openrouter_stream_sse_accumulation():
    """OpenRouter SSE: split content + fragmented tool_call arguments assemble
    into one ModelResult; usage from the final chunk is captured."""
    import json as _json

    from harness.llm.provider import OpenRouterProvider

    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": '{"comm'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'and": "ls"}'}}]}}
            ]
        },
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
    ]
    lines = [f"data: {_json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    class StubStreamResp:
        is_success = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(lines)

    class StubClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, json, params):
            assert json["stream"] is True
            assert json["stream_options"] == {"include_usage": True}
            assert params == {}
            return StubStreamResp()

    prov = OpenRouterProvider(Config().provider)
    prov._client = lambda: StubClient()
    deltas, res = _drain(
        prov.stream("m", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    )
    assert "".join(deltas) == "Hello"
    assert res.text == "Hello"
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'
    assert res.tokens_in == 11 and res.tokens_out == 7


def _spec_names(h):
    return [s["function"]["name"] for s in h.tools.tool_specs()]


def _htools(**tool_kwargs):
    """Harness with in-memory repo + tool/hook kwargs injected via the
    constructor (tools=[...] / hooks=[...])."""
    cfg = dataclasses.replace(Config(), database_url="")
    return Harness(
        cfg, system_prompt="sys.", provider=FakeProvider(context_window=4000), **tool_kwargs
    )


def test_builtin_tools_enabled_by_default():
    h = _htools()
    from harness.tools.builtin import default_tools

    assert set(_spec_names(h)) == {t.name for t in default_tools()}


def test_no_tools_via_false_or_empty():
    """None => all built-ins; False or [] => no tools at all."""
    assert _spec_names(_htools(tools=False)) == []
    assert _spec_names(_htools(tools=[])) == []


def test_tools_true_means_all_builtins():
    """tools=True == tools=None (all built-ins). Pinned because the annotation
    accepts bool: before this was handled, True type-checked but crashed."""
    from harness.tools.builtin import default_tools

    assert set(_spec_names(_htools(tools=True))) == {t.name for t in default_tools()}


def test_tools_list_is_the_selection():
    """The tools list IS the selection: only listed tools appear in the specs,
    and an omitted built-in isn't runnable (falls through to the MCP-index
    lookup, which reports it unknown)."""
    from harness.tools.builtin import Bash, SearchTools

    h = _htools(tools=[Bash(), SearchTools()])
    s = h.start_session("u1")
    assert set(_spec_names(h)) == {"Bash", "SearchTools"}
    out = h.tools.dispatch(s, {"id": "x", "function": {"name": "RenderUI", "arguments": "{}"}})
    assert "Unknown tool" in out["content"]


def test_custom_tool_injected_and_dispatched():
    """A developer tool passed via tools= appears in the specs and its handler
    runs on dispatch, receiving the parsed arguments."""
    from harness.tools.builtin import default_tools, make_tool

    seen = {}

    def handler(session, args):
        seen["args"] = args
        return f"echo:{args.get('q', '')}"

    tool = make_tool(
        "MySearch",
        "Search my index.",
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler,
    )
    h = _htools(tools=[*default_tools(), tool])
    s = h.start_session("u1")
    assert "MySearch" in _spec_names(h)
    out = h.tools.dispatch(
        s, {"id": "x", "function": {"name": "MySearch", "arguments": '{"q": "hello"}'}}
    )
    assert out["content"] == "echo:hello"
    assert seen["args"] == {"q": "hello"}


def test_duplicate_tool_name_raises():
    from harness.tools.builtin import Bash, make_tool

    dup = make_tool("Bash", "nope", {"type": "object", "properties": {}}, lambda s, a: "")
    try:
        _htools(tools=[Bash(), dup])
    except ValueError as e:
        assert "Bash" in str(e)
    else:
        raise AssertionError("expected ValueError for duplicate tool name")


def test_hooks_fire_and_transform():
    """Hooks fire at each lifecycle point in list order; before_tool/after_tool
    can transform the args and result."""
    from harness.core.loop import Hook
    from harness.tools.builtin import default_tools

    events = []

    class Recorder(Hook):
        def __init__(self, tag):
            self.tag = tag

        def before_turn(self, session, message):
            events.append((self.tag, "before_turn", message))

        def after_turn(self, session, result):
            events.append((self.tag, "after_turn", result.status))

        def before_tool(self, session, name, args):
            events.append((self.tag, "before_tool", name))

    class Rewrite(Hook):
        def before_tool(self, session, name, args):
            return {"command": "echo rewritten"}  # transform the call

        def after_tool(self, session, name, result):
            return result + "\n[audited]"  # transform the result

    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "Bash", "arguments": '{"command": "echo orig"}'}}
        ]
    )
    p.queue(content="done")
    cfg = dataclasses.replace(Config(), database_url="")
    h = Harness(
        cfg,
        system_prompt="sys.",
        provider=p,
        tools=default_tools(),
        hooks=[Recorder("a"), Rewrite()],
    )
    s = h.start_session("u1")
    r = h.run_turn(s, "go")

    # before_turn fired once, after_turn fired with the final status
    assert ("a", "before_turn", "go") in events
    assert ("a", "after_turn", "ok") in events
    # before_tool observed the Bash call
    assert ("a", "before_tool", "Bash") in events
    # Rewrite.before_tool changed the command; Rewrite.after_tool annotated output
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert "rewritten" in tool_turns[0].content["content"]
    assert "orig" not in tool_turns[0].content["content"]
    assert "[audited]" in tool_turns[0].content["content"]
    assert r.status == "ok"


def test_prompt_guidance_reflects_active_tools():
    """The composed prompt includes guidance only for active tools (a custom
    tool's guidance appears; an omitted built-in's does not)."""
    from harness.tools.builtin import SearchTools, make_tool

    custom = make_tool(
        "Weather",
        "Get weather.",
        {"type": "object", "properties": {}},
        lambda s, a: "sunny",
        guidance="- Weather(): fetch the local forecast.",
    )
    # No system_prompt override, so the prompt is composed from the active tools.
    cfg = dataclasses.replace(Config(), database_url="")
    h = Harness(cfg, provider=FakeProvider(context_window=4000), tools=[SearchTools(), custom])
    sp = h.loop.system_prompt
    assert "fetch the local forecast" in sp  # custom tool guidance present
    assert "per-session sandbox" not in sp  # Bash omitted -> its guidance absent


def test_detect_provider_precedence():
    """azure_endpoint wins over openrouter_api_key; neither => FakeProvider.
    detect_provider is the opt-in CLI/demo heuristic; Harness never calls it."""
    from harness.llm.provider import (
        AzureFoundryProvider,
        FakeProvider,
        OpenRouterProvider,
        detect_provider,
        provider_label,
    )
    from harness.settings import ProviderConfig

    az = Config(
        provider=ProviderConfig(
            azure_endpoint="https://r.services.ai.azure.com",
            azure_api_key="k",
            openrouter_api_key="sk-or-x",
        )
    )
    assert isinstance(detect_provider(az), AzureFoundryProvider)
    assert provider_label(az) == "Azure AI Foundry"

    orr = Config(provider=ProviderConfig(azure_endpoint="", openrouter_api_key="sk-or-x"))
    assert isinstance(detect_provider(orr), OpenRouterProvider)
    assert provider_label(orr) == "OpenRouter"

    none = Config(provider=ProviderConfig(azure_endpoint="", openrouter_api_key=""))
    assert isinstance(detect_provider(none), FakeProvider)
    assert provider_label(none) == "FakeProvider (offline)"

    # Vertex (project set) beats OpenRouter; Bedrock (region set) beats OpenRouter.
    from harness.llm.provider import BedrockProvider, VertexProvider

    vx = Config(provider=ProviderConfig(vertex_project="p", openrouter_api_key="sk-or-x"))
    assert isinstance(detect_provider(vx), VertexProvider)
    assert provider_label(vx) == "Google Vertex AI"

    br = Config(provider=ProviderConfig(bedrock_region="us-east-1", openrouter_api_key="sk-or-x"))
    assert isinstance(detect_provider(br), BedrockProvider)
    assert provider_label(br) == "AWS Bedrock"


def test_bedrock_openai_to_converse_translation():
    """OpenAI-shaped messages/tools map to the Converse schema: system split out,
    consecutive same-role turns merged, tool_calls -> toolUse, tool -> toolResult."""
    from harness.llm.provider import BedrockProvider
    from harness.settings import ProviderConfig

    prov = BedrockProvider(ProviderConfig(bedrock_region="us-east-1"))
    system, conv = prov._to_converse(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": "sure",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]
    )
    assert system == [{"text": "be terse"}]
    assert conv[0] == {"role": "user", "content": [{"text": "run ls"}]}
    assert conv[1]["role"] == "assistant"
    tu = conv[1]["content"][1]["toolUse"]
    assert tu == {"toolUseId": "t1", "name": "Bash", "input": {"command": "ls"}}
    # tool result + following user turn merge into ONE user message (alternation).
    assert conv[2]["role"] == "user"
    assert conv[2]["content"][0]["toolResult"]["toolUseId"] == "t1"
    assert conv[2]["content"][-1] == {"text": "thanks"}

    tcfg = prov._tool_config([
        {"type": "function", "function": {"name": "Bash", "description": "run", "parameters": {"type": "object"}}}
    ])
    spec = tcfg["tools"][0]["toolSpec"]
    assert spec["name"] == "Bash" and spec["inputSchema"] == {"json": {"type": "object"}}


class _StubBedrockRuntime:
    """Minimal boto3 bedrock-runtime stand-in for complete()/converse_stream()."""

    def __init__(self, converse=None, stream_events=None):
        self._converse = converse
        self._events = stream_events or []
        self.calls = {}

    def converse(self, **kwargs):
        self.calls["converse"] = kwargs
        return self._converse

    def converse_stream(self, **kwargs):
        self.calls["converse_stream"] = kwargs
        return {"stream": iter(self._events)}


def test_bedrock_complete_translates_response():
    """Converse response (text + toolUse + usage) -> OpenAI-shaped ModelResult."""
    from harness.llm.provider import BedrockProvider
    from harness.settings import ProviderConfig

    prov = BedrockProvider(ProviderConfig(bedrock_region="us-east-1"))
    prov._runtime = _StubBedrockRuntime(
        converse={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "hi"},
                        {"toolUse": {"toolUseId": "t9", "name": "Bash", "input": {"command": "ls"}}},
                    ],
                }
            },
            "usage": {"inputTokens": 11, "outputTokens": 7},
        }
    )
    res = prov.complete("anthropic.claude", [{"role": "user", "content": "yo"}], tools=None)
    assert res.text == "hi"
    tc = res.tool_calls[0]
    assert tc["id"] == "t9" and tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'  # dict re-serialized to JSON string
    assert res.tokens_in == 11 and res.tokens_out == 7


def test_bedrock_stream_assembles_text_and_tool_calls():
    """converse_stream events -> yielded text deltas + assembled tool_calls + usage."""
    from harness.llm.provider import BedrockProvider
    from harness.settings import ProviderConfig

    prov = BedrockProvider(ProviderConfig(bedrock_region="us-east-1"))
    prov._runtime = _StubBedrockRuntime(
        stream_events=[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "Bash"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"command":'}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ' "ls"}'}}}},
            {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 4}}},
        ]
    )
    deltas, res = _drain(
        prov.stream("m", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    )
    assert "".join(deltas) == "Hello"
    assert res.text == "Hello"
    tc = res.tool_calls[0]
    assert tc["id"] == "t1" and tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'
    assert res.tokens_in == 3 and res.tokens_out == 4


def test_build_provider_requires_explicit_name():
    """build_provider(cfg) never sniffs config contents — it's a pure
    name -> factory lookup and raises if cfg.provider.name is unset."""
    from harness.llm.provider import (
        AzureFoundryProvider,
        FakeProvider,
        OpenRouterProvider,
        build_provider,
    )
    from harness.settings import ProviderConfig

    try:
        build_provider(Config())
    except ValueError as e:
        assert "cfg.provider.name" in str(e)
    else:
        raise AssertionError("expected ValueError when cfg.provider.name is unset")

    assert isinstance(
        build_provider(Config(provider=ProviderConfig(name="fake"))), FakeProvider
    )
    assert isinstance(
        build_provider(
            Config(provider=ProviderConfig(name="openrouter", openrouter_api_key="sk-x"))
        ),
        OpenRouterProvider,
    )
    assert isinstance(
        build_provider(
            Config(provider=ProviderConfig(name="azure", azure_endpoint="https://x", azure_api_key="k"))
        ),
        AzureFoundryProvider,
    )


def test_azure_complete_api_key_and_params():
    """AzureFoundryProvider.complete: api-key header auth, api-version query
    param, /openai/v1 base, and OpenAI-shaped response parsing."""
    from harness.llm.provider import AzureFoundryProvider
    from harness.settings import ProviderConfig

    captured: dict = {}

    class StubResp:
        is_success = True

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }

    class StubClient:
        def __init__(self, headers):
            captured["headers"] = headers

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json, params):
            captured["url"] = url
            captured["params"] = params
            captured["payload"] = json
            return StubResp()

    cfg = ProviderConfig(
        azure_endpoint="https://r.services.ai.azure.com/",
        azure_api_key="secret-key",
        azure_api_version="preview",
    )
    prov = AzureFoundryProvider(cfg)
    # verify the client is built with api-key auth + /openai/v1 base
    prov._client = lambda: StubClient(prov._auth_headers())

    res = prov.complete(
        "my-deployment", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    )

    assert captured["headers"] == {"api-key": "secret-key"}
    assert captured["url"] == "/chat/completions"
    assert captured["params"] == {"api-version": "preview"}
    assert captured["payload"]["model"] == "my-deployment"
    assert captured["payload"]["tool_choice"] == "auto"
    assert res.text == "hi there"
    assert res.tokens_in == 5 and res.tokens_out == 2


def test_azure_stream_reuses_openai_base():
    """AzureFoundryProvider inherits the OpenAI-compatible SSE assembly:
    split content + fragmented tool_call arguments merge into one ModelResult,
    and api-version params are forwarded to the streaming request."""
    import json as _json

    from harness.llm.provider import AzureFoundryProvider
    from harness.settings import ProviderConfig

    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": '{"comm'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'and": "ls"}'}}]}}
            ]
        },
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
    ]
    lines = [f"data: {_json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    captured: dict = {}

    class StubStreamResp:
        is_success = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(lines)

    class StubClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, json, params):
            captured["params"] = params
            assert json["stream"] is True
            return StubStreamResp()

    cfg = ProviderConfig(
        azure_endpoint="https://r.services.ai.azure.com",
        azure_api_key="k",
        azure_api_version="2024-10-21",
    )
    prov = AzureFoundryProvider(cfg)
    prov._client = lambda: StubClient()
    deltas, res = _drain(
        prov.stream("m", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    )
    assert "".join(deltas) == "Hello"
    assert res.text == "Hello"
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "Bash"
    assert tc["function"]["arguments"] == '{"command": "ls"}'
    assert res.tokens_in == 11 and res.tokens_out == 7
    assert captured["params"] == {"api-version": "2024-10-21"}
