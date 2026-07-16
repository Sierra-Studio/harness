"""The `@tool` decorator: a plain function becomes a `Tool`.

`make_tool` already builds a Tool without subclassing, but it makes the
developer hand-write the JSON-Schema and adapt to the `(session, args)`
handler signature. This decorator closes that gap: the schema is inferred
from the function's type hints and defaults, descriptions come from the
docstring (Google-style ``Args:`` section), and arguments arrive as real
parameters.

    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        \"\"\"Current weather for a city.

        Args:
            city: City name to look up.
            unit: "celsius" or "fahrenheit".

        Guidance:
            Prefer this over Bash+curl when the user asks about weather.
        \"\"\"
        ...

Precedence is always: explicit decorator option > docstring/signature
inference. `name=` beats the function name, `description=` beats the
docstring summary, `parameters=` beats hint inference, `guidance=` beats
the ``Guidance:`` section.

The decorated object IS a `Tool` (drop it in `tools=[...]` or a `ToolBundle`)
and stays callable with the original signature, so tests and other code can
keep invoking it directly. Parameters named `ctx` / `session` are excluded
from the model-facing schema and injected at dispatch time.

Inference is deliberately shallow — plain types, `list[T]`, `Literal`,
`Optional`. For anything richer, pass an explicit ``parameters=`` schema
(same shape `make_tool` takes) or subclass `Tool`.
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from collections.abc import Callable

from ..models import Session
from .builtin import Tool, ToolContext

_INJECTED = ("ctx", "session")

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _schema_for(annotation) -> dict:
    """Best-effort JSON-Schema fragment for one annotation. Unknown -> {} (any)."""
    if annotation is inspect.Parameter.empty or annotation is typing.Any:
        return {}
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        values = list(typing.get_args(annotation))
        base = _TYPE_MAP.get(type(values[0]))
        out: dict = {"enum": values}
        if base:
            out["type"] = base
        return out
    if origin in (typing.Union, getattr(types, "UnionType", None)):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        # Optional[T] -> schema of T; wider unions -> any.
        return _schema_for(args[0]) if len(args) == 1 else {}
    if origin in (list, tuple, set, frozenset):
        item_args = typing.get_args(annotation)
        items = _schema_for(item_args[0]) if item_args else {}
        return {"type": "array", "items": items} if items else {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if isinstance(annotation, type) and annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    return {}


# Google-style section headers we understand; any of them also terminates the
# section before it. Args/Arguments/Parameters and Guidance are consumed;
# the rest are recognized only so they don't bleed into a preceding section.
_SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Guidance|Returns|Yields|Raises|Examples?|Notes?):\s*$",
    flags=re.MULTILINE,
)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str], str]:
    """Split a docstring into (description, {param: description}, guidance).

    The description is everything before the first Google-style section
    header; ``Args:`` lines look like ``name: text`` (continuations indented
    further); a ``Guidance:`` section becomes the tool's guidance snippet.
    """
    if not doc:
        return "", {}, ""
    doc = inspect.cleandoc(doc)
    headers = list(_SECTION_RE.finditer(doc))
    if not headers:
        return doc.strip(), {}, ""
    description = doc[: headers[0].start()].strip()
    sections: dict[str, str] = {}
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(doc)
        sections[m.group(1)] = doc[m.end() : end]

    params: dict[str, str] = {}
    current = None
    args_body = sections.get("Args") or sections.get("Arguments") or sections.get("Parameters") or ""
    for line in args_body.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):  # left the indented block
            break
        pm = re.match(r"\s+(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
        if pm:
            current = pm.group(1).lstrip("*")
            params[current] = pm.group(2).strip()
        elif current:
            params[current] += " " + line.strip()

    guidance = " ".join(sections.get("Guidance", "").split())
    return description, params, guidance


def _infer_parameters(fn: Callable) -> dict:
    """Build the JSON-Schema object for `fn`'s model-facing parameters."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs -> fall back to raw annotations
        hints = {}
    _, param_docs, _ = _parse_docstring(fn.__doc__ or "")

    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in _INJECTED:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"@tool cannot infer a schema for *args/**kwargs on {fn.__name__}(); "
                "pass an explicit parameters= schema instead."
            )
        prop = _schema_for(hints.get(name, param.annotation))
        if name in param_docs:
            prop = {**prop, "description": param_docs[name]}
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class FunctionTool(Tool):
    """A `Tool` wrapping a plain function; also callable as the original.

    Dispatch maps the model's args dict onto the function's signature and
    injects `ctx` / `session` if the function declares them. Non-string
    return values are JSON-encoded so structured results need no boilerplate.
    """

    def __init__(self, fn: Callable, name: str, description: str, parameters: dict, guidance: str):
        self._fn = fn
        self._wants = tuple(k for k in _INJECTED if k in inspect.signature(fn).parameters)
        self.name = name
        self.description = description
        self.parameters = parameters
        self.guidance = guidance
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def run(self, ctx: ToolContext, session: Session, args: dict) -> str:
        known = set(self.parameters.get("properties", {}))
        kwargs = {k: v for k, v in args.items() if k in known}
        if "ctx" in self._wants:
            kwargs["ctx"] = ctx
        if "session" in self._wants:
            kwargs["session"] = session
        out = self._fn(**kwargs)
        if out is None:
            return ""
        if isinstance(out, str):
            return out
        return json.dumps(out, ensure_ascii=False, default=str)


def tool(
    fn: Callable | None = None,
    *,
    name: str = "",
    description: str = "",
    parameters: dict | None = None,
    guidance: str = "",
) -> FunctionTool | Callable[[Callable], FunctionTool]:
    """Decorate a function as a `Tool`. Usable bare (`@tool`) or with options.

    Defaults: `name` from the function name, `description` from the docstring
    (text before the first section header), `parameters` inferred from type
    hints and defaults with per-argument descriptions from the ``Args:``
    section, `guidance` from a ``Guidance:`` section. Any explicit option
    overrides its inferred counterpart wholesale.
    """

    def deco(f: Callable) -> FunctionTool:
        tool_name = name or f.__name__
        desc, _, doc_guidance = _parse_docstring(f.__doc__ or "")
        tool_desc = description or desc
        if not tool_desc:
            raise TypeError(
                f"@tool needs a description for {f.__name__}(): add a docstring "
                "or pass description=."
            )
        params = parameters if parameters is not None else _infer_parameters(f)
        return FunctionTool(f, tool_name, tool_desc, params, guidance or doc_guidance)

    return deco(fn) if fn is not None else deco


__all__ = ["tool", "FunctionTool"]
