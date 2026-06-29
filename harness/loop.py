"""The agent loop: perceive -> build context -> call model -> run tools ->
repeat, under a per-session token-budget guard that stops and returns the
partial response when the ceiling is reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from .config import Config
from .memory import Memory
from .models import Session
from .observer import Observer
from .prompt import skills_block, with_today
from .provider import Provider
from .repository import Repository
from .tools import ToolRegistry


@dataclass
class TurnResult:
    text: str
    status: str       # ok | budget_exhausted | max_steps
    steps: int
    tokens_spent: int


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
    result: Optional[TurnResult] = None


class AgentLoop:
    def __init__(self, cfg: Config, repo: Repository, provider: Provider,
                 memory: Memory, tools: ToolRegistry, observer: Observer,
                 system_prompt: str):
        self.cfg = cfg
        self.repo = repo
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.observer = observer
        self.system_prompt = system_prompt

    def start_session(self, external_id: str, model: str = "") -> Session:
        user = self.repo.get_or_create_user(external_id)
        model = model or self.cfg.model
        ctx = self.provider.model_context_window(model)
        return self.repo.create_session(
            user.id, model, ctx, self.cfg.token_budget_per_session)

    def run_turn(self, session: Session, user_message: Any) -> TurnResult:
        """Run a turn to completion and return its TurnResult.

        Thin consumer of run_turn_stream so the synchronous API is preserved for
        callers (and tests) that don't care about intermediate events.
        """
        result: Optional[TurnResult] = None
        for ev in self.run_turn_stream(session, user_message):
            if ev.kind == "final":
                result = ev.result
        # run_turn_stream always emits a final event before returning.
        assert result is not None
        return result

    def run_turn_stream(self, session: Session,
                        user_message: Any) -> Iterator[LoopEvent]:
        session = self.repo.get_session(session.id)
        self.memory.append(session, "user", user_message)
        self.memory.maybe_checkpoint(session)

        # Prompt layers, ordered for cache friendliness:
        #   [ global system prompt ]  <- identical across users; cacheable prefix
        #   [ this user's skills   ]  <- per-user catalog (name + summary only)
        #   [ today's date         ]  <- volatile; appended last so the rest stays stable
        # The per-user catalog is re-read each turn so skills added mid-session
        # appear immediately. Bodies are NOT injected — loaded via GetSkill.
        catalog = skills_block(self.repo.list_skills(session.user_id),
                               self.cfg.skills_in_prompt_limit)
        base = self.system_prompt + (f"\n\n{catalog}" if catalog else "")
        prompt = with_today(base)

        final_text = ""
        for step in range(self.cfg.max_steps):
            session = self.repo.get_session(session.id)

            # --- token-budget guard: stop & return partial (budget 0 = unlimited) ---
            if session.token_budget and session.tokens_spent >= session.token_budget:
                self.repo.set_session_status(session.id, "budget_exhausted")
                yield LoopEvent("final", result=TurnResult(
                    final_text or "(token budget exhausted)",
                    "budget_exhausted", step, session.tokens_spent))
                return

            messages = self.memory.build_window(session, prompt)
            # Stream the model turn: drive provider.stream(), surfacing each text
            # delta as a "text" event; the assembled ModelResult is the generator's
            # return value (StopIteration.value).
            with self.observer.timed(session.id, None, "model_call",
                                     {"step": step}) as slot:
                gen = self.provider.stream(session.model, messages,
                                           self.tools.builtin_specs())
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
                session, "assistant", res.message,
                tokens_in=res.tokens_in, tokens_out=res.tokens_out)
            self.repo.add_session_tokens(session.id, res.tokens_in + res.tokens_out)
            session = self.repo.get_session(session.id)

            self.memory.maybe_summarize(session, prompt)

            if not res.tool_calls:
                yield LoopEvent("final", result=TurnResult(
                    res.text, "ok", step + 1, session.tokens_spent))
                return

            final_text = res.text or final_text
            for call in res.tool_calls:
                name, args, call_id = self.tools._parse(call)
                yield LoopEvent("tool_start", name=name, args=args, call_id=call_id)
                out = self.tools.dispatch(session, call)
                self.memory.append(session, "tool", {
                    "tool_call_id": out["tool_call_id"], "content": out["content"]})
                self.observer.log(session.id, assistant_turn.id, "tool_call",
                                  {"name": out["name"]})
                yield LoopEvent("tool_result", name=out["name"],
                                call_id=out["tool_call_id"], content=out["content"])

        yield LoopEvent("final", result=TurnResult(
            final_text or "(max steps reached)", "max_steps", self.cfg.max_steps,
            self.repo.get_session(session.id).tokens_spent))
