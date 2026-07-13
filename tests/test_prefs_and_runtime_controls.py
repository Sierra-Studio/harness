"""Unit tests for harness.interfaces.prefs (local TUI/CLI preferences) and the
Harness runtime-control methods (set_persona / set_session_model /
set_session_budget) that back the /persona, /model, /budget slash commands.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from harness.interfaces import prefs
from harness.settings import Config
from harness.testing import offline_harness


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(prefs, "STORE", tmp_path / "preferences.json")
    return prefs.STORE


def test_load_returns_defaults_when_missing(temp_store):
    d = prefs.load()
    assert d["theme"] == "" and d["persona"] == "" and d["token_budget"] is None


def test_save_merges_rather_than_overwrites(temp_store):
    prefs.save(theme="nord", model="gpt-5")
    prefs.save(persona="Atlas")
    d = prefs.load()
    assert d == {
        "theme": "nord",
        "persona": "Atlas",
        "system_prompt": "",
        "model": "gpt-5",
        "token_budget": None,
        "response_reserve_tokens": None,
        "permission_mode": "",
    }


def test_save_ignores_unknown_keys(temp_store):
    prefs.save(theme="nord", bogus="ignored")
    assert "bogus" not in prefs.load()


def test_apply_defaults_uses_saved_prefs_as_fallback(temp_store, monkeypatch):
    monkeypatch.delenv("HARNESS_MODEL", raising=False)
    monkeypatch.delenv("TOKEN_BUDGET_PER_SESSION", raising=False)
    prefs.save(model="gpt-5", token_budget=0)
    cfg = prefs.apply_defaults(Config())
    assert cfg.provider.model == "gpt-5"
    assert cfg.loop.token_budget_per_session == 0


def test_permission_mode_default_and_env():
    from harness.settings import PermissionConfig

    assert PermissionConfig.from_env({}).mode == "auto"  # default: all commands approved
    assert PermissionConfig.from_env({"HARNESS_PERMISSION_MODE": "manual"}).mode == "manual"
    assert PermissionConfig.from_env({"HARNESS_PERMISSION_MODE": "bogus"}).mode == "auto"


def test_permission_mode_saved_pref_applies(temp_store, monkeypatch):
    monkeypatch.delenv("HARNESS_PERMISSION_MODE", raising=False)
    prefs.save(permission_mode="manual")
    cfg = prefs.apply_defaults(Config())
    assert cfg.permissions.mode == "manual"


def test_permission_mode_real_env_wins_over_pref(temp_store, monkeypatch):
    # A real shell env var outranks a saved preference: apply_defaults must NOT
    # overwrite the env-derived mode with the pref. Config() defaults to "auto".
    monkeypatch.setenv("HARNESS_PERMISSION_MODE", "auto")
    prefs.save(permission_mode="manual")
    cfg = prefs.apply_defaults(Config(), from_dotenv=frozenset())
    assert cfg.permissions.mode == "auto"


def test_apply_defaults_real_env_var_wins_over_prefs(temp_store, monkeypatch):
    prefs.save(model="gpt-5")
    monkeypatch.setenv("HARNESS_MODEL", "env-model")
    cfg = prefs.apply_defaults(Config.from_env())
    assert cfg.provider.model == "env-model"


def test_apply_defaults_dotenv_value_does_not_block_saved_pref(temp_store, monkeypatch):
    """A key populated by _load_dotenv (passed via from_dotenv) must NOT act
    like a real env var — a saved preference should still win over it. This
    guards the precedence bug where a repo's checked-in .env permanently
    shadowed any /model or /budget the user set."""
    monkeypatch.setenv("HARNESS_MODEL", "dotenv-model")  # simulates _load_dotenv's setdefault
    prefs.save(model="gpt-5")
    cfg = prefs.apply_defaults(Config.from_env(), from_dotenv={"HARNESS_MODEL"})
    assert cfg.provider.model == "gpt-5"


def test_apply_defaults_leaves_cfg_untouched_when_no_prefs_saved(temp_store):
    cfg = Config()
    assert prefs.apply_defaults(cfg) is cfg  # identity fast-path, no dataclasses.replace


def test_harness_set_persona_rebuilds_prompt_and_resets():
    h = offline_harness()
    default_prompt = h.loop.system_prompt
    h.set_persona(persona="You are Atlas, a terse SRE.")
    assert "Atlas" in h.loop.system_prompt
    assert h.loop.system_prompt != default_prompt
    h.set_persona()  # no args resets to the default identity
    assert h.loop.system_prompt == default_prompt


def test_harness_set_persona_system_prompt_overrides_persona():
    h = offline_harness()
    h.set_persona(persona="ignored", system_prompt="RAW OVERRIDE")
    assert h.loop.system_prompt == "RAW OVERRIDE"


def test_harness_set_session_model_refreshes_context_window():
    h = offline_harness()
    session = h.start_session("u1")
    h.set_session_model(session, "some/other-model")
    assert session.model == "some/other-model"
    assert session.context_window == h.provider.model_context_window("some/other-model")


def test_harness_set_session_budget_clamps_negative_to_zero():
    h = offline_harness()
    session = h.start_session("u1")
    h.set_session_budget(session, 12345)
    assert session.token_budget == 12345
    h.set_session_budget(session, -5)
    assert session.token_budget == 0


def test_parse_budget_vocabulary():
    from harness.interfaces import ui

    for word in ("0", "none", "unlimited", "inf", "-1", "NONE", "Unlimited"):
        assert ui.parse_budget(word) == 0
    assert ui.parse_budget("200000") == 200000
    assert ui.parse_budget("not-a-number") is None


def test_is_query_guards_against_misread_subcommands():
    """Regression: `/persona show` and `/persona help` were silently SETTING
    the persona to the literal text "show"/"help" (with a false "updated and
    saved" success message) instead of displaying it — a completely
    reasonable guess at the syntax was corrupting saved state silently."""
    from harness.interfaces import ui

    assert ui.is_query(["show"]) is True
    assert ui.is_query(["help"]) is True
    assert ui.is_query(["SHOW"]) is True  # case-insensitive
    assert ui.is_query(["?"]) is True
    assert ui.is_query([]) is False  # callers already handle `not args` themselves
    assert ui.is_query(["clear"]) is False  # clear has its own explicit branch
    assert ui.is_query(["You", "are", "Atlas"]) is False  # real multi-word content
    assert ui.is_query(["show", "me"]) is False  # not a bare query word


def test_parse_command_never_raises_on_unbalanced_quotes():
    """Regression: `shlex.split` raises ValueError on any unmatched quote
    character, and a plain apostrophe in ordinary prose ("you're", "don't",
    "it's") is exactly that — an unmatched quote. Before this fix, typing
    `/system-prompt you're a helpful bot` raised an uncaught ValueError that
    crashed the entire `harness chat` process. parse_command must swallow
    that and fall back to a plain whitespace split instead."""
    from harness.interfaces import ui

    cmd, args, raw = ui.parse_command("/system-prompt you're a helpful bot")
    assert cmd == "/system-prompt"
    assert args == ["you're", "a", "helpful", "bot"]  # whitespace-split fallback, no crash
    assert raw == "you're a helpful bot"

    cmd, args, raw = ui.parse_command("/persona don't be verbose, it's important")
    assert raw == "don't be verbose, it's important"


def test_parse_command_raw_is_verbatim_not_lossy_rejoined():
    """`raw` must be the exact original text (not `" ".join(args)`, which
    collapses repeated whitespace and can't survive a tokenization fallback)."""
    from harness.interfaces import ui

    _, _, raw = ui.parse_command("/persona   extra   spaces   preserved")
    assert raw == "extra   spaces   preserved"


def test_parse_command_still_shlex_tokenizes_quoted_args():
    """Commands that DO want shell-like tokenization (/skills add, /mcp
    stdio) must still get quoted multi-word tokens as one arg when the line
    parses cleanly."""
    from harness.interfaces import ui

    cmd, args, _ = ui.parse_command('/skills add greet "say hi"')
    assert cmd == "/skills"
    assert args == ["add", "greet", "say hi"]


def test_parse_command_empty_remainder():
    from harness.interfaces import ui

    cmd, args, raw = ui.parse_command("/persona")
    assert cmd == "/persona" and args == [] and raw == ""
