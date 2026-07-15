"""Unit tests for the Permissions gate (core/permissions.py).

Pins the documented promises that keep manual mode usable: read-only tools
never prompt, a missing asker never hangs a non-interactive run, ALLOW is
one-shot while ALWAYS is sticky, and unknown modes are ignored.
"""

from __future__ import annotations

from harness.core import ALLOW, ALWAYS, DENY
from harness.core.permissions import READONLY, Permissions


def test_manual_mode_without_asker_allows():
    # No human to ask (piped / non-interactive): blocking would hang the turn,
    # so the documented fallback is allow.
    p = Permissions(mode="manual", asker=None)
    assert p.check("Bash", {"command": "rm -rf /tmp/x"}) is True


def test_readonly_tools_never_prompt_in_manual_mode():
    def exploding_asker(name, args):
        raise AssertionError(f"asker must not be consulted for read-only tool {name}")

    p = Permissions(mode="manual", asker=exploding_asker)
    for name in READONLY:
        assert p.check(name, {}) is True


def test_allow_is_one_shot_but_always_is_sticky():
    decisions = iter([ALLOW, DENY, ALWAYS])
    asked: list[str] = []

    def asker(name, args):
        asked.append(name)
        return next(decisions)

    p = Permissions(mode="manual", asker=asker)
    assert p.check("Bash", {}) is True  # ALLOW grants this call only...
    assert p.check("Bash", {}) is False  # ...so the next call asks again (DENY)
    assert p.check("Bash", {}) is True  # ALWAYS grants...
    assert p.check("Bash", {}) is True  # ...and is remembered for the session
    assert asked == ["Bash", "Bash", "Bash"]  # the 4th check never asked


def test_always_is_remembered_per_tool_not_globally():
    p = Permissions(mode="manual", asker=lambda name, args: ALWAYS)
    p.check("Bash", {})
    asked: list[str] = []
    p.asker = lambda name, args: (asked.append(name), DENY)[1]
    assert p.check("Bash", {}) is True  # still granted, no prompt
    assert p.check("Write", {}) is False  # a different tool prompts (and is denied)
    assert asked == ["Write"]


def test_auto_mode_never_consults_asker():
    def exploding_asker(name, args):
        raise AssertionError("auto mode must bypass the asker")

    p = Permissions(mode="auto", asker=exploding_asker)
    assert p.check("Bash", {"command": "echo hi"}) is True


def test_unknown_mode_falls_back_to_auto_and_set_mode_ignores_unknown():
    p = Permissions(mode="bogus")
    assert p.mode == "auto"  # constructor sanitizes
    assert p.set_mode("bogus") == "auto"  # unknown value ignored, current kept
    assert p.set_mode("manual") == "manual"
    assert p.set_mode("nonsense") == "manual"  # still manual


def test_toggle_flips_between_auto_and_manual():
    p = Permissions(mode="auto")
    assert p.toggle() == "manual"
    assert p.toggle() == "auto"
