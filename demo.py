"""Offline, network-free demo of the whole harness.

Uses the in-memory repository and a scripted FakeProvider, so it runs with no
Postgres and no OpenRouter key. It exercises: the loop with a Bash tool call,
per-session token-budget stop, chained summarization, user-turn checkpoints,
multi-tenant isolation, and skill induction.

    python3 demo.py
"""
from __future__ import annotations

import dataclasses

from harness.app import Harness
from harness.config import Config
from harness.provider import FakeProvider


class DemoProvider(FakeProvider):
    """Deterministic provider that also induces a skill and classifies subjects."""

    def classify_subject(self, model, messages):
        return "deployment questions"

    def induce_skills(self, model, signals):
        return [{
            "name": "deploy-checklist",
            "summary": "Run tests, then deploy and verify",
            "body": "1) run the test suite\n2) deploy\n3) check health endpoint",
        }]


def section(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main() -> None:
    # tight budgets so summarization & the budget guard actually fire in a demo
    cfg = dataclasses.replace(
        Config(),
        database_url="",  # offline demo always uses the in-memory repo
        token_budget_per_session=500_000,
        response_reserve_tokens=10,
        checkpoint_every_user_turns=3,
        summary_keep_ratio=0.2,
        max_steps=6,
    )
    provider = DemoProvider(context_window=160)  # small window -> forces summaries
    h = Harness(cfg, system_prompt="You are a test agent.", echo=True,
                provider=provider)

    # ---------------------------------------------------------------
    section("1 · LOOP WITH A TOOL CALL (Bash in the sandbox)")
    s = h.start_session("alice")
    provider.queue(tool_calls=[{
        "id": "c1",
        "function": {"name": "Bash", "arguments": '{"command": "echo hello-from-sandbox"}'},
    }])
    provider.queue(content="The sandbox printed: hello-from-sandbox.")
    r = h.run_turn(s, "Run echo hello-from-sandbox and report the output.")
    print(f"-> assistant: {r.text}  (status={r.status}, steps={r.steps})")

    # ---------------------------------------------------------------
    section("2 · CHECKPOINTS + CHAINED SUMMARIZATION")
    for i in range(2, 7):
        provider.queue(content=f"Reply #{i} about deployments and rollbacks " + "x" * 80)
        r = h.run_turn(s, f"Question {i}: how do I handle deployment step {i}? " + "y" * 60)
        print(f"-> turn {i}: status={r.status} tokens_spent={r.tokens_spent}")
    print(f"\nsummaries created: {len(h.repo.summaries)}")
    print(f"checkpoints: {[c['label'] + '@' + str(c['at_user_turn']) for c in h.repo.checkpoints]}")
    print(f"active turns in window: {len(h.repo.active_turns(s.id))} "
          f"(of {len([t for t in h.repo.turns if t.session_id == s.id])} total)")

    # ---------------------------------------------------------------
    section("3 · PER-SESSION TOKEN BUDGET -> STOP & RETURN PARTIAL")
    tight = dataclasses.replace(cfg, token_budget_per_session=30)
    p2 = DemoProvider(context_window=4000)
    h2 = Harness(tight, system_prompt="You are a test agent.", provider=p2)
    s2 = h2.start_session("alice")
    # provider keeps asking for tools, so the loop would run forever w/o the guard
    for _ in range(10):
        p2.queue(content="partial...", tool_calls=[{
            "id": "x", "function": {"name": "Bash", "arguments": '{"command": "echo k"}'}}])
    r = h2.run_turn(s2, "do something expensive")
    print(f"-> status={r.status} (expected budget_exhausted), "
          f"tokens_spent={r.tokens_spent}, budget={s2.token_budget}")

    # ---------------------------------------------------------------
    section("4 · MULTI-TENANT ISOLATION")
    sb = h.start_session("bob")
    provider.queue(content="Hi Bob, this is a fresh context.")
    h.run_turn(sb, "Hello, who am I talking to?")
    alice_id = h.repo.get_or_create_user("alice").id
    bob_id = h.repo.get_or_create_user("bob").id
    alice_turns = [t for t in h.repo.turns if t.user_id == alice_id]
    bob_turns = [t for t in h.repo.turns if t.user_id == bob_id]
    print(f"alice turns: {len(alice_turns)} · bob turns: {len(bob_turns)}")
    print("bob cannot see alice's turns: "
          f"{all(t.user_id == bob_id for t in bob_turns)}")

    # ---------------------------------------------------------------
    section("5 · SKILL INDUCTION (every N sessions)")
    print("closing sessions for alice until the induction cadence is hit...")
    created = h.close_session(s)
    # open/close more sessions to reach the cadence (default 10)
    for _ in range(cfg.skill_induction_every_sessions):
        ss = h.start_session("alice")
        created = h.close_session(ss)
        if created:
            break
    print(f"closed sessions for alice: {h.repo.count_closed_sessions(alice_id)}")
    print(f"induced skills: {[(s.name, s.origin) for s in h.repo.list_skills(alice_id)]}")

    # ---------------------------------------------------------------
    section("6 · OBSERVABILITY (step logs)")
    by_type: dict[str, int] = {}
    ti = to = 0
    for log in h.repo.step_logs:
        by_type[log["step_type"]] = by_type.get(log["step_type"], 0) + 1
        ti += log["tokens_in"] or 0
        to += log["tokens_out"] or 0
    print(f"step counts by type: {by_type}")
    print(f"total tokens logged: in={ti} out={to}")
    print("\nDemo complete. ✔")


if __name__ == "__main__":
    main()
