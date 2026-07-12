# Skills

Skills are reusable, user-owned procedures surfaced via the `SearchSkills` /
`GetSkill` tools and grown automatically by **induction**.

`Skills` is pluggable the same way `Repository`/`Provider`/`SandboxBackend`
are — inject an implementation via `Harness(skills=...)`:

```python
class Skills(abc.ABC):
    def list(self, user_id: str) -> list[Skill]: ...
    def search(self, user_id: str, query: str, k: int) -> list[Skill]: ...
    def get(self, user_id: str, name: str) -> Skill | None: ...
    def add(self, user_id, name, summary, body, origin) -> Skill: ...
    def on_session_closed(self, session: Session) -> list[str]: ...
```

Two implementations ship:

- **`RepositorySkills`** (default) — persists through the injected
  `Repository` and induces new skills via the injected `Provider`.
- **`NullSkills`** — turns the feature off entirely: `add()` raises
  `NotImplementedError` (refuses to persist rather than silently no-op), and
  `on_session_closed()` never triggers induction.

## Induction

Every `SKILL_INDUCTION_EVERY_SESSIONS` (default 10) closed sessions,
`RepositorySkills` gathers recent signals from the session and asks the
provider's `induce_skills()` to mine recurring request patterns into candidate
skills. Candidates are deduped by embedding before being persisted, so the same
underlying procedure doesn't accumulate near-duplicate skills over time.

## In the prompt

Up to `MemoryConfig.skills_in_prompt_limit` (default 30) of a user's skills are
rendered directly into the system prompt (`skills_block()` in
`harness/memory/persona.py`). Beyond that limit, the model reaches the long tail via
`SearchSkills` instead — bounded prompt cost regardless of how many skills a
user accumulates.
