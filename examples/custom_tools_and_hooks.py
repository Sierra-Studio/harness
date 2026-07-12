"""Compose the tool set and attach lifecycle hooks from the outside — no env
vars, no editing the library.

    uv run python examples/custom_tools_and_hooks.py

Shows three things:
  1. Importing the built-in tools and composing your own `tools=` list
     (built-ins + a custom tool). Include a tool to have it, omit it to not.
  2. A custom tool built with `make_tool` (handler is `(session, args) -> str`).
  3. A `Hook` that intercepts every tool call (audit + a tiny guardrail).

Runs fully offline against the built-in FakeProvider (no API key needed).
"""

from __future__ import annotations

import dataclasses

from harness import Bash, Harness, Hook, SearchTools, make_tool
from harness.llm.provider import FakeProvider
from harness.settings import Config


# --- 1. a custom tool: (session, args) -> str, no subclassing needed ----------
def _greet(session, args) -> str:
    return f"Hello, {args.get('name', 'world')}!"


greet = make_tool(
    "Greet",
    "Greet someone by name.",
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    _greet,
    guidance="- Greet(name): return a friendly greeting.",
)


# --- 2. a hook: intercept + audit every tool call -----------------------------
class AuditHook(Hook):
    def before_turn(self, session, message):
        print(f"[hook] turn started: {message!r}")

    def before_tool(self, session, name, args):
        print(f"[hook] calling {name} with {args}")
        # tiny guardrail: block a dangerous command by rewriting it
        if name == "Bash" and "rm -rf" in args.get("command", ""):
            return {"command": "echo 'blocked by guardrail'"}
        return None  # leave args unchanged

    def after_tool(self, session, name, result):
        return result + "\n[audited]"  # annotate every tool result

    def after_turn(self, session, result):
        print(f"[hook] turn ended: status={result.status}")


def main() -> None:
    # Offline provider scripted to call Greet, then Bash, then answer.
    provider = FakeProvider(context_window=4000)
    provider.queue(
        tool_calls=[{"id": "c1", "function": {"name": "Greet", "arguments": '{"name": "Ada"}'}}]
    )
    provider.queue(
        tool_calls=[
            {"id": "c2", "function": {"name": "Bash", "arguments": '{"command": "rm -rf /"}'}}
        ]
    )
    provider.queue(content="All done.")

    cfg = dataclasses.replace(Config(), database_url="")  # in-memory repo
    # Compose the tool set: a subset of built-ins + our custom tool. RenderUI,
    # CallTool, the skill tools, etc. are deliberately omitted. To keep every
    # built-in, pass tools=[*default_tools(), greet] instead.
    h = Harness(
        cfg,
        provider=provider,
        tools=[SearchTools(), Bash(), greet],
        hooks=[AuditHook()],
    )

    print("Active tools:", [s["function"]["name"] for s in h.tools.tool_specs()])

    s = h.start_session("demo-user")
    result = h.run_turn(s, "greet Ada, then try to wipe the disk")
    print("\nFinal:", result.text)


if __name__ == "__main__":
    main()
