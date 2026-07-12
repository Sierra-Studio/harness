"""Tests for the ToolProvider composition SPI — run with: python3 -m pytest -q."""

from __future__ import annotations

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.settings import Config
from harness.tools.builtin import make_tool
from harness.tools.capabilities import ProviderHost, ToolBundle, ToolProvider


def _harness(providers, **overrides):
    cfg = dataclasses.replace(Config(), database_url="", **overrides)
    return Harness(
        cfg, system_prompt="sys.", provider=FakeProvider(context_window=4000), tools=providers
    )


def _tool(name):
    return make_tool(name, f"{name} desc", {"type": "object", "properties": {}},
                     lambda s, a: name, guidance=f"- {name}: does {name}")


def _names(h):
    return {s["function"]["name"] for s in h.tools.tool_specs()}


def test_tool_bundle_registers_its_tools():
    h = _harness([ToolBundle([_tool("alpha"), _tool("beta")])])
    assert {"alpha", "beta"} <= _names(h)


def test_multiple_providers_compose():
    h = _harness([ToolBundle([_tool("alpha")]), ToolBundle([_tool("beta")])])
    assert {"alpha", "beta"} <= _names(h)


def test_provider_guidance_reaches_prompt():
    # providers register BEFORE prompt assembly, so guidance is advertised.
    cfg = dataclasses.replace(Config(), database_url="")
    h = Harness(
        cfg,
        persona="you are x",
        provider=FakeProvider(context_window=4000),
        tools=[ToolBundle([_tool("alpha")])],
    )
    assert "alpha" in h.loop.system_prompt


class _Boom(ToolProvider):
    optional = False

    def register(self, host: ProviderHost) -> None:
        raise RuntimeError("cannot connect")


class _BoomOptional(_Boom):
    optional = True


def test_required_provider_failure_aborts_construction():
    try:
        _harness([_Boom()])
        assert False, "expected construction to raise"
    except RuntimeError as e:
        assert "cannot connect" in str(e)


def test_optional_provider_failure_degrades():
    # optional provider that fails to register is skipped; others still load.
    h = _harness([ToolBundle([_tool("alpha")]), _BoomOptional()])
    assert "alpha" in _names(h)


def test_close_stops_providers():
    stopped = []

    class _Tracked(ToolProvider):
        def register(self, host):
            host.add_tool(_tool("gamma"))

        def stop(self):
            stopped.append(True)

    h = _harness([_Tracked()])
    assert "gamma" in _names(h)
    h.close()
    assert stopped == [True]
