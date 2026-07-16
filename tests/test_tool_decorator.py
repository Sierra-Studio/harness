"""@tool decorator: schema inference, docstring parsing, dispatch integration."""

from __future__ import annotations

from typing import Literal

import pytest
from helpers import make_harness

from harness.tools import FunctionTool, Tool, ToolBundle, tool


def _props(t):
    return t.parameters["properties"]


def test_bare_decorator_infers_everything():
    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Current weather for a city.

        Args:
            city: City name to look up.
            unit: Temperature unit.
        """
        return f"{city}:{unit}"

    assert isinstance(get_weather, Tool)
    assert isinstance(get_weather, FunctionTool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Current weather for a city."
    assert _props(get_weather)["city"] == {"type": "string", "description": "City name to look up."}
    assert _props(get_weather)["unit"]["type"] == "string"
    assert get_weather.parameters["required"] == ["city"]
    # still callable as the plain function
    assert get_weather("SP") == "SP:celsius"


def test_type_inference_covers_common_annotations():
    @tool(description="d")
    def f(
        a: int,
        b: float,
        c: bool,
        d: list[str],
        e: dict,
        g: Literal["x", "y"],
        h: int | None = None,
        i=None,
    ):
        return ""

    p = _props(f)
    assert p["a"] == {"type": "integer"}
    assert p["b"] == {"type": "number"}
    assert p["c"] == {"type": "boolean"}
    assert p["d"] == {"type": "array", "items": {"type": "string"}}
    assert p["e"] == {"type": "object"}
    assert p["g"] == {"enum": ["x", "y"], "type": "string"}
    assert p["h"] == {"type": "integer"}
    assert p["i"] == {}  # unannotated -> any
    assert f.parameters["required"] == ["a", "b", "c", "d", "e", "g"]


def test_overrides_win_over_inference():
    schema = {"type": "object", "properties": {"z": {"type": "string"}}, "required": []}

    @tool(name="Renamed", description="Custom.", parameters=schema, guidance="Use wisely.")
    def f(anything: int):
        """Ignored docstring."""
        return ""

    assert f.name == "Renamed"
    assert f.description == "Custom."
    assert f.parameters is schema
    assert f.guidance == "Use wisely."
    assert f.spec()["function"]["name"] == "Renamed"


def test_guidance_from_docstring_section():
    @tool
    def f(x: int):
        """Do the thing.

        Args:
            x: The input.

        Guidance:
            Prefer this over Bash when the user
            asks for the thing.
        """
        return ""

    assert f.description == "Do the thing."
    assert f.guidance == "Prefer this over Bash when the user asks for the thing."
    assert _props(f)["x"]["description"] == "The input."


def test_guidance_section_order_does_not_matter():
    @tool
    def f(x: int):
        """Do the thing.

        Guidance:
            Use sparingly.

        Args:
            x: The input.
        """
        return ""

    assert f.description == "Do the thing."
    assert f.guidance == "Use sparingly."
    assert _props(f)["x"]["description"] == "The input."


def test_decorator_guidance_overrides_docstring_section():
    @tool(guidance="Explicit wins.")
    def f():
        """Do the thing.

        Guidance:
            Ignored.
        """
        return ""

    assert f.guidance == "Explicit wins."


def test_missing_description_raises():
    with pytest.raises(TypeError, match="description"):

        @tool
        def f(x: int):
            return ""


def test_var_args_rejected():
    with pytest.raises(TypeError, match="parameters="):

        @tool(description="d")
        def f(*args):
            return ""


def test_run_injects_ctx_and_session_and_encodes_json():
    @tool(description="d")
    def f(x: int, ctx=None, session=None):
        return {"x": x, "has_ctx": ctx is not None, "has_session": session is not None}

    out = f.run("CTX", "SESSION", {"x": 3, "junk": "dropped"})
    assert out == '{"x": 3, "has_ctx": true, "has_session": true}'
    # ctx/session never leak into the model-facing schema
    assert set(_props(f)) == {"x"}


def test_run_returns_str_verbatim_and_none_as_empty():
    @tool(description="d")
    def s():
        return "plain"

    @tool(description="d")
    def n():
        return None

    assert s.run(None, None, {}) == "plain"
    assert n.run(None, None, {}) == ""


def test_decorated_tool_dispatches_through_harness():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    h = make_harness(tools=[ToolBundle([add])])
    s = h.start_session("u1")
    assert "add" in [t["function"]["name"] for t in h.tools.tool_specs()]
    out = h.tools.dispatch(s, {"id": "1", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}})
    assert out["content"] == "5"
    h.close()
