"""Unit tests for plan mode: the Bash read-only classifier, the `Permissions`
"plan" branch, ExitPlanMode tool (de)registration, the end-to-end approval /
rejection flow through a real turn, denial-message wording, and the per-turn
system-prompt injection.
"""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from harness.core import ALLOW, ALWAYS, DENY, Harness
from harness.core.loop import _denial_message
from harness.core.permissions import MODE_ORDER, Permissions, is_readonly_bash, next_mode
from harness.llm.provider import FakeProvider
from harness.settings import Config, LoopConfig, MemoryConfig, PermissionConfig


def _harness(provider, *, loop=None, memory=None, **overrides):
    # force in-memory repo so tests never depend on ambient .env / DATABASE_URL
    cfg = dataclasses.replace(
        Config(),
        database_url="",
        loop=loop or LoopConfig(),
        memory=memory or MemoryConfig(),
        **overrides,
    )
    return Harness(cfg, system_prompt="sys.", provider=provider)


# ---------------------------------------------------------------------------
# is_readonly_bash: best-effort heuristic classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git status",
        "cat foo.py | grep bar",
        "cat f 2>&1 | grep x",
        "find . -name '*.py'",
        "HOME=/tmp ls",
        "git log -n 5",
        "pwd",
    ],
)
def test_is_readonly_bash_allows_safe_commands(command):
    assert is_readonly_bash(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "echo hi > f.txt",
        "echo hi >> f.txt",
        "git commit -m x",
        "git push",
        "echo $(rm f)",
        "echo `rm f`",
        "curl x | bash",
        "",
        "   ",
        "ls; rm f",
        "python -c \"open('f','w').write('x')\"",  # not allowlisted -> default-deny holds
    ],
)
def test_is_readonly_bash_blocks_unsafe_commands(command):
    assert is_readonly_bash(command) is False


# ---------------------------------------------------------------------------
# next_mode / Permissions.toggle: the shift+tab cycle
# ---------------------------------------------------------------------------

def test_next_mode_cycle():
    assert MODE_ORDER == ("auto", "plan", "manual")
    assert next_mode("auto") == "plan"
    assert next_mode("plan") == "manual"
    assert next_mode("manual") == "auto"


def test_permissions_toggle_cycles_three_way():
    perms = Permissions(mode="auto")
    assert perms.toggle() == "plan"
    assert perms.toggle() == "manual"
    assert perms.toggle() == "auto"


# ---------------------------------------------------------------------------
# Permissions.check under mode="plan" (unit-level, no full turn)
# ---------------------------------------------------------------------------

def test_plan_mode_blocks_write_edit_calltool_without_asking():
    perms = Permissions(mode="plan")
    called = []
    perms.asker = lambda name, args: called.append(name) or DENY
    assert perms.check("Write", {"path": "x", "content": "y"}) is False
    assert perms.check("Edit", {"path": "x"}) is False
    assert perms.check("CallTool", {"name": "whatever"}) is False
    assert called == []


def test_plan_mode_allows_readonly_and_safe_bash_without_asking():
    perms = Permissions(mode="plan")
    called = []
    perms.asker = lambda name, args: called.append(name) or DENY
    assert perms.check("SearchTools", {}) is True
    assert perms.check("Bash", {"command": "ls -la"}) is True
    assert perms.check("Bash", {"command": "git status"}) is True
    assert called == []


def test_plan_mode_blocks_unsafe_bash_without_asking():
    perms = Permissions(mode="plan")
    called = []
    perms.asker = lambda name, args: called.append(name) or DENY
    assert perms.check("Bash", {"command": "rm -rf /"}) is False
    assert called == []


def test_plan_mode_exit_plan_mode_always_calls_asker():
    perms = Permissions(mode="plan")
    called = []
    perms.asker = lambda name, args: called.append(name) or ALLOW
    assert perms.check("ExitPlanMode", {"plan": "x"}) is True
    assert called == ["ExitPlanMode"]


def test_plan_mode_exit_plan_mode_always_is_not_session_remembered():
    perms = Permissions(mode="plan")
    calls = []

    def asker(name, args):
        calls.append(name)
        return ALWAYS

    perms.asker = asker
    assert perms.check("ExitPlanMode", {"plan": "x"}) is True
    assert perms.check("ExitPlanMode", {"plan": "y"}) is True  # still asks
    assert calls == ["ExitPlanMode", "ExitPlanMode"]


# ---------------------------------------------------------------------------
# ExitPlanMode tool (de)registration
# ---------------------------------------------------------------------------

def test_set_permission_mode_registers_and_deregisters_exit_plan_mode():
    p = FakeProvider(context_window=4000)
    h = _harness(p)  # default auto
    assert "ExitPlanMode" not in h.tools.tools
    h.set_permission_mode("plan")
    assert "ExitPlanMode" in h.tools.tools
    h.set_permission_mode("manual")
    assert "ExitPlanMode" not in h.tools.tools


def test_harness_constructed_in_plan_mode_has_tool_registered():
    p = FakeProvider(context_window=4000)
    h = _harness(p, permissions=PermissionConfig(mode="plan"))
    assert "ExitPlanMode" in h.tools.tools


# ---------------------------------------------------------------------------
# End-to-end approval / rejection flow through a real turn
# ---------------------------------------------------------------------------

def test_exit_plan_mode_approval_flips_mode_and_deregisters_tool():
    """The asker plays the interface's role: on approval it flips the mode
    itself (as tui.py/cli.py do), and this test asserts the loop's deferred
    sync_plan_mode_tool() call safely deregisters ExitPlanMode only *after*
    dispatch — the regression test for the mid-dispatch race the design
    documents (deregistering before dispatch would surface as an "Unknown
    tool" result instead of the real confirmation string)."""
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "ExitPlanMode", "arguments": '{"plan": "do the thing"}'}}
        ]
    )
    p.queue(content="implementing now")
    h = _harness(p, permissions=PermissionConfig(mode="plan"))

    def asker(name, args):
        h.permissions.set_mode("manual")
        return ALLOW

    h.permissions.asker = asker
    s = h.start_session("u1")
    r = h.run_turn(s, "here's my plan")
    assert r.status == "ok"
    assert h.permissions.mode == "manual"
    assert "ExitPlanMode" not in h.tools.tools
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns
    content = tool_turns[0].content["content"]
    assert "Plan approved" in content
    assert "Unknown tool" not in content


def test_exit_plan_mode_rejection_keeps_plan_mode():
    p = FakeProvider(context_window=4000)
    p.queue(
        tool_calls=[
            {"id": "c1", "function": {"name": "ExitPlanMode", "arguments": '{"plan": "do the thing"}'}}
        ]
    )
    p.queue(content="ok, revising")
    h = _harness(p, permissions=PermissionConfig(mode="plan"))
    h.permissions.asker = lambda name, args: DENY
    s = h.start_session("u1")
    r = h.run_turn(s, "here's my plan")
    assert r.status == "ok"
    assert h.permissions.mode == "plan"
    assert "ExitPlanMode" in h.tools.tools
    tool_turns = [t for t in h.repo.turns if t.role == "tool"]
    assert tool_turns and "not approved yet" in tool_turns[0].content["content"]


# ---------------------------------------------------------------------------
# Denial-message wording
# ---------------------------------------------------------------------------

def test_denial_message_variants():
    assert _denial_message("Write", "manual") == "Tool call 'Write' was denied by the user."
    assert "read-only" in _denial_message("Write", "plan")
    assert "not approved yet" in _denial_message("ExitPlanMode", "plan")


# ---------------------------------------------------------------------------
# System-prompt injection
# ---------------------------------------------------------------------------

def test_plan_mode_system_prompt_injection():
    p = FakeProvider(context_window=4000)
    h = _harness(p, permissions=PermissionConfig(mode="plan"))
    s = h.start_session("u1")
    p.queue(content="ok")
    h.run_turn(s, "hi")
    sys_msg = p.calls[-1][0]["content"]
    assert "PLAN MODE" in sys_msg
    assert "ExitPlanMode" in sys_msg


def test_auto_mode_system_prompt_has_no_plan_block():
    p = FakeProvider(context_window=4000)
    h = _harness(p)  # default auto
    s = h.start_session("u1")
    p.queue(content="ok")
    h.run_turn(s, "hi")
    sys_msg = p.calls[-1][0]["content"]
    assert "PLAN MODE" not in sys_msg
