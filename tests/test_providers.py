"""Tests for the ToolProvider composition SPI — run with: python3 -m pytest -q."""

from __future__ import annotations

import dataclasses

from helpers import make_harness

from harness.core import Harness
from harness.llm.provider import FakeProvider
from harness.settings import Config
from harness.tools.builtin import make_tool
from harness.tools.capabilities import ProviderHost, ToolBundle, ToolProvider


def _harness(providers, **overrides):
    return make_harness(tools=providers, **overrides)


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
        raise AssertionError("expected construction to raise")
    except RuntimeError as e:
        assert "cannot connect" in str(e)


def test_optional_provider_failure_degrades():
    # optional provider that fails to register is skipped; others still load.
    h = _harness([ToolBundle([_tool("alpha")]), _BoomOptional()])
    assert "alpha" in _names(h)


class _FakeResp:
    def __init__(self, status, body, ok=False):
        self.is_success = ok
        self.status_code = status
        self._body = body

        class _Req:
            url = "https://x/openai/v1/chat/completions"

        self.request = _Req()

    def read(self):  # streamed responses must be read before .json()
        pass

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def test_check_surfaces_content_filter_reason():
    from harness.llm.provider import OpenAICompatibleProvider as P

    body = {
        "error": {
            "code": "content_filter",
            "message": "blocked",
            "innererror": {
                "content_filter_result": {
                    "violence": {"filtered": True, "severity": "medium"},
                    "hate": {"filtered": False},
                }
            },
        }
    }
    try:
        P._check(_FakeResp(400, body))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "content filter" in str(e).lower() and "violence" in str(e)
        assert "rephrase" in str(e).lower()


def test_check_surfaces_generic_message_and_bare_status():
    from harness.llm.provider import OpenAICompatibleProvider as P

    try:
        P._check(_FakeResp(400, {"error": {"message": "bad api-version"}}))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "bad api-version" in str(e)
    try:
        P._check(_FakeResp(500, None))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "500" in str(e)
    P._check(_FakeResp(200, {}, ok=True))  # success does not raise


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
