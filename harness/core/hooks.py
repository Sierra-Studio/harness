"""Hook decorators: a plain function becomes a single-point `Hook`.

`Hook` covers four interception points; subclassing is right when one object
carries state across several of them, but the common case is one function for
one point (block a command, redact a result, audit a turn). These decorators
close that gap the same way `@tool` does for tools:

    @before_tool
    def block_rm(session, name, args):
        if name == "Bash" and "rm -rf" in args.get("command", ""):
            return {"command": "echo blocked"}

    Harness(cfg, provider=llm, hooks=[block_rm])

The decorated object IS a `Hook` (drop it in `hooks=[...]`) and stays callable
with the original signature, so tests can keep invoking it directly. Return
semantics are the `Hook` method's: `before_tool` may return a replacement args
dict, `after_tool` a replacement result string; `before_turn`/`after_turn`
returns are ignored.

The function must accept the hook point's positional arguments — checked at
decoration time, so a wrong signature fails at import, not mid-turn.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from .loop import Hook

# Hook point -> the positional arguments the loop passes (after self).
_HOOK_SIGS: dict[str, tuple[str, ...]] = {
    "before_turn": ("session", "message"),
    "after_turn": ("session", "result"),
    "before_tool": ("session", "name", "args"),
    "after_tool": ("session", "name", "result"),
}


class FunctionHook(Hook):
    """A `Hook` with exactly one point filled by a plain function.

    The function is stored as an instance attribute under the hook-point name,
    shadowing the class no-op — the loop's `h.before_tool(...)` call finds it
    there and invokes it unbound (no `self`), which is exactly the plain
    function's signature.
    """

    def __init__(self, fn: Callable, point: str):
        expected = _HOOK_SIGS[point]
        try:
            inspect.signature(fn).bind(*(object(),) * len(expected))
        except TypeError:
            raise TypeError(
                f"@{point} function {fn.__name__}() must accept "
                f"({', '.join(expected)}); got signature {inspect.signature(fn)}."
            ) from None
        setattr(self, point, fn)
        self.point = point
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def before_turn(fn: Callable) -> FunctionHook:
    """`fn(session, message)` runs once when a turn starts."""
    return FunctionHook(fn, "before_turn")


def after_turn(fn: Callable) -> FunctionHook:
    """`fn(session, result)` runs once when a turn ends, with its TurnResult."""
    return FunctionHook(fn, "after_turn")


def before_tool(fn: Callable) -> FunctionHook:
    """`fn(session, name, args)` runs before each tool call; return a dict to
    replace the args, or None to leave them unchanged."""
    return FunctionHook(fn, "before_tool")


def after_tool(fn: Callable) -> FunctionHook:
    """`fn(session, name, result)` runs after each tool call; return a string
    to replace the result, or None to leave it unchanged."""
    return FunctionHook(fn, "after_tool")


__all__ = ["FunctionHook", "before_turn", "after_turn", "before_tool", "after_tool"]
