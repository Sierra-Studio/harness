"""The agent loop: perceive -> build context -> call model -> run tools ->
repeat, under a per-session token-budget guard that stops and returns the
partial response when the ceiling is reached.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..llm import Provider
from ..memory import Memory, Skills, skills_block, with_today
from ..models import Session
from ..observability import Observer
from ..persistence import Repository
from ..settings import Config
from ..tools import ToolRegistry


@dataclass
class TurnResult:
    text: str
    status: str  # ok | budget_exhausted | max_steps | tool_limit_exhausted
    steps: int
    tokens_spent: int


class Hook:
    """Interception points around a turn and each tool call.

    Subclass and override only the methods you need; the rest are no-ops. Hooks
    are the interception/transform + cross-cutting seam (guardrails, redaction,
    audit) — they fire on every turn regardless of how the caller runs it.
    Pure observation is better served by iterating `run_turn_stream`'s events.
    """

    def before_turn(self, session: Session, message: Any) -> None:
        """Called once when a turn starts, after the user message is recorded."""

    def after_turn(self, session: Session, result: TurnResult) -> None:
        """Called once when a turn ends, with its final TurnResult."""

    def before_tool(self, session: Session, name: str, args: dict) -> dict | None:
        """Called before a tool runs. Return a replacement args dict to transform
        the call, or None to leave the args unchanged."""

    def after_tool(self, session: Session, name: str, result: str) -> str | None:
        """Called after a tool runs. Return a replacement result string, or None
        to leave the result unchanged."""


@dataclass
class LoopEvent:
    """A single observable event emitted while a turn runs.

    kind:
      "text"        incremental assistant content  -> text
      "tool_start"  a tool is about to run         -> name, args, call_id
      "tool_result" a tool finished                -> name, content, call_id
      "final"       the turn ended                 -> result (TurnResult)
    """

    kind: str
    text: str = ""
    name: str = ""
    call_id: str = ""
    args: dict = field(default_factory=dict)
    content: str = ""
    result: TurnResult | None = None


class AgentLoop:
    def __init__(
        self,
        cfg: Config,
        repo: Repository,
        provider: Provider,
        memory: Memory,
        tools: ToolRegistry,
        observer: Observer,
        system_prompt: str,
        skills: Skills,
        hooks: list[Hook] | None = None,
    ):
        self.cfg = cfg
        self.repo = repo
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.observer = observer
        self.system_prompt = system_prompt
        self.skills = skills
        self.hooks = list(hooks or ())

    # ---- hook fan-out (a raising hook never kills the turn) ----
    def _fire_turn_hook(self, method: str, session: Session, payload: Any) -> None:
        for h in self.hooks:
            try:
                getattr(h, method)(session, payload)
            except Exception as e:  # noqa: BLE001 - hooks must not break the loop
                self.observer.log(session.id, None, "hook_error", {"hook": method, "error": str(e)})

    def _fire_before_tool(self, session: Session, name: str, args: dict) -> dict:
        for h in self.hooks:
            try:
                new = h.before_tool(session, name, args)
                if isinstance(new, dict):
                    args = new
            except Exception as e:  # noqa: BLE001
                self.observer.log(
                    session.id, None, "hook_error", {"hook": "before_tool", "error": str(e)}
                )
        return args

    def _fire_after_tool(self, session: Session, name: str, result: str) -> str:
        for h in self.hooks:
            try:
                new = h.after_tool(session, name, result)
                if isinstance(new, str):
                    result = new
            except Exception as e:  # noqa: BLE001
                self.observer.log(
                    session.id, None, "hook_error", {"hook": "after_tool", "error": str(e)}
                )
        return result

    def start_session(self, external_id: str, model: str = "") -> Session:
        user = self.repo.get_or_create_user(external_id)
        model = model or self.cfg.provider.model
        ctx = self.provider.model_context_window(model)
        return self.repo.create_session(user.id, model, ctx, self.cfg.loop.token_budget_per_session)

    def run_turn(self, session: Session, user_message: Any) -> TurnResult:
        """Run a turn to completion and return its TurnResult.

        Thin consumer of run_turn_stream so the synchronous API is preserved for
        callers (and tests) that don't care about intermediate events.
        """
        result: TurnResult | None = None
        for ev in self.run_turn_stream(session, user_message):
            if ev.kind == "final":
                result = ev.result
        # run_turn_stream always emits a final event before returning.
        assert result is not None
        return result

    def run_turn_stream(self, session: Session, user_message: Any) -> Iterator[LoopEvent]:
        session = self.repo.get_session(session.id)
        self.memory.append(session, "user", user_message)
        self.memory.maybe_checkpoint(session)
        self._fire_turn_hook("before_turn", session, user_message)

        # Prompt layers, ordered for cache friendliness:
        #   [ global system prompt ]  <- identical across users; cacheable prefix
        #   [ this user's skills   ]  <- per-user catalog (name + summary only)
        #   [ today's date         ]  <- volatile; appended last so the rest stays stable
        # The per-user catalog is re-read each turn so skills added mid-session
        # appear immediately. Bodies are NOT injected — loaded via GetSkill.
        catalog = skills_block(
            self.skills.list(session.user_id), self.cfg.memory.skills_in_prompt_limit
        )
        base = self.system_prompt + (f"\n\n{catalog}" if catalog else "")
        prompt = with_today(base)

        final_text = ""
        tool_calls_this_turn = 0
        for step in range(self.cfg.loop.max_steps):
            session = self.repo.get_session(session.id)

            # --- token-budget guard: stop & return partial (budget 0 = unlimited) ---
            if session.token_budget and session.tokens_spent >= session.token_budget:
                self.repo.set_session_status(session.id, "budget_exhausted")
                result = TurnResult(
                    final_text or "(token budget exhausted)",
                    "budget_exhausted",
                    step,
                    session.tokens_spent,
                )
                self._fire_turn_hook("after_turn", session, result)
                yield LoopEvent("final", result=result)
                return

            messages = self.memory.build_window(session, prompt)
            # Stream the model turn: drive provider.stream(), surfacing each text
            # delta as a "text" event; the assembled ModelResult is the generator's
            # return value (StopIteration.value).
            with self.observer.timed(session.id, None, "model_call", {"step": step}) as slot:
                gen = self.provider.stream(session.model, messages, self.tools.tool_specs())
                while True:
                    try:
                        delta = next(gen)
                    except StopIteration as stop:
                        res = stop.value
                        break
                    if delta:
                        yield LoopEvent("text", text=delta)
                slot["tokens_in"] = res.tokens_in
                slot["tokens_out"] = res.tokens_out

            assistant_turn = self.memory.append(
                session,
                "assistant",
                res.message,
                tokens_in=res.tokens_in,
                tokens_out=res.tokens_out,
            )
            self.repo.add_session_tokens(session.id, res.tokens_in + res.tokens_out)
            session = self.repo.get_session(session.id)

            self.memory.maybe_summarize(session, prompt)

            if not res.tool_calls:
                result = TurnResult(res.text, "ok", step + 1, session.tokens_spent)
                self._fire_turn_hook("after_turn", session, result)
                yield LoopEvent("final", result=result)
                return

            final_text = res.text or final_text
            calls = res.tool_calls
            per_step_limit = self.cfg.loop.max_tool_calls_per_step
            if per_step_limit and len(calls) > per_step_limit:
                self.observer.log(
                    session.id,
                    assistant_turn.id,
                    "tool_calls_truncated",
                    {"requested": len(calls), "kept": per_step_limit},
                )
                calls = calls[:per_step_limit]

            for call in calls:
                per_turn_limit = self.cfg.loop.max_tool_calls_per_turn
                if per_turn_limit and tool_calls_this_turn >= per_turn_limit:
                    result = TurnResult(
                        final_text or "(tool call limit reached)",
                        "tool_limit_exhausted",
                        step + 1,
                        self.repo.get_session(session.id).tokens_spent,
                    )
                    self._fire_turn_hook("after_turn", session, result)
                    yield LoopEvent("final", result=result)
                    return
                tool_calls_this_turn += 1
                name, args, call_id = self.tools._parse(call)
                # before_tool hooks may transform the args; dispatch the effective call.
                args = self._fire_before_tool(session, name, args)
                yield LoopEvent("tool_start", name=name, args=args, call_id=call_id)
                out = self.tools.dispatch(
                    session, {"id": call_id, "function": {"name": name, "arguments": args}}
                )
                content = self._fire_after_tool(session, name, out["content"])
                self.memory.append(
                    session,
                    "tool",
                    {"tool_call_id": out["tool_call_id"], "content": content},
                )
                self.observer.log(session.id, assistant_turn.id, "tool_call", {"name": out["name"]})
                yield LoopEvent(
                    "tool_result",
                    name=out["name"],
                    call_id=out["tool_call_id"],
                    content=content,
                )

        result = TurnResult(
            final_text or "(max steps reached)",
            "max_steps",
            self.cfg.loop.max_steps,
            self.repo.get_session(session.id).tokens_spent,
        )
        self._fire_turn_hook("after_turn", session, result)
        yield LoopEvent("final", result=result)
