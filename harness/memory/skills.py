"""Skills: reusable, user-owned procedures surfaced via SearchSkills/GetSkill
and grown by induction.

Pluggable the same way `Repository`/`Provider`/`SandboxBackend` are: inject a
`Skills` implementation via `Harness(skills=...)`. The static default,
`RepositorySkills`, persists through the already-injected `Repository` and
induces new skills via the already-injected `Provider` — never anything
`Harness` selects by inspecting `Config`. Pass `skills=NullSkills()` to
disable the feature entirely (empty catalog, no induction, no persistence)
without touching your `Repository` at all.
"""

from __future__ import annotations

import abc

# These classes define a method named `list`, which shadows the builtin inside
# the class bodies — annotations there must say `builtins.list` explicitly or
# mypy resolves them to the method and stops type-checking the whole seam.
import builtins

from ..llm import Provider
from ..models import Session, Skill
from ..observability import Observer
from ..persistence import Repository
from ..settings import Config


class Skills(abc.ABC):
    """Read/write + induction contract used by the loop, the SearchSkills/
    GetSkill tools, and the per-user prompt catalog."""

    @abc.abstractmethod
    def list(self, user_id: str) -> builtins.list[Skill]: ...

    @abc.abstractmethod
    def search(self, user_id: str, query: str, k: int) -> builtins.list[Skill]: ...

    @abc.abstractmethod
    def get(self, user_id: str, name: str) -> Skill | None: ...

    @abc.abstractmethod
    def add(self, user_id: str, name: str, summary: str, body: str, origin: str) -> Skill: ...

    @abc.abstractmethod
    def on_session_closed(self, session: Session) -> builtins.list[str]:
        """Called by `Harness.close_session`. Returns the names of any newly
        induced skills; a no-op implementation just returns `[]`."""


class NullSkills(Skills):
    """Disables the feature: empty catalog, no induction, no persistence.

    SearchSkills/GetSkill still work (they always return "nothing found"
    rather than erroring), so you don't need to also drop those tools from
    `tools=` — this is the backend switch, same relationship `sandbox=` has
    to the `Bash` tool.
    """

    def list(self, user_id: str) -> builtins.list[Skill]:
        return []

    def search(self, user_id: str, query: str, k: int) -> builtins.list[Skill]:
        return []

    def get(self, user_id: str, name: str) -> Skill | None:
        return None

    def add(self, user_id: str, name: str, summary: str, body: str, origin: str) -> Skill:
        raise NotImplementedError(
            "Skills are disabled (NullSkills) — pass a real Skills implementation "
            "(e.g. RepositorySkills) to persist one."
        )

    def on_session_closed(self, session: Session) -> builtins.list[str]:
        return []


class RepositorySkills(Skills):
    """Default `Skills` implementation: persists through the injected
    `Repository` and induces new skills via the injected `Provider`, every
    `cfg.memory.skill_induction_every_sessions` closed sessions (deduped by
    name). This is exactly today's built-in behavior — just behind the
    `Skills` seam so it can be swapped out."""

    def __init__(self, repo: Repository, provider: Provider, cfg: Config, observer: Observer):
        self.repo = repo
        self.provider = provider
        self.cfg = cfg
        self.observer = observer

    def list(self, user_id: str) -> builtins.list[Skill]:
        return self.repo.list_skills(user_id)

    def search(self, user_id: str, query: str, k: int) -> builtins.list[Skill]:
        return self.repo.search_skills(user_id, query, k)

    def get(self, user_id: str, name: str) -> Skill | None:
        return self.repo.get_skill(user_id, name)

    def add(self, user_id: str, name: str, summary: str, body: str, origin: str) -> Skill:
        return self.repo.add_skill(user_id, name, summary, body, origin)

    def on_session_closed(self, session: Session) -> builtins.list[str]:
        """Trigger induction when the user hits a multiple of the cadence.
        Returns the names of any newly created skills."""
        user_id = session.user_id
        closed = self.repo.count_closed_sessions(user_id)
        if closed == 0 or closed % self.cfg.memory.skill_induction_every_sessions != 0:
            return []

        signals = self._gather_signals(session)
        drafts = self.provider.induce_skills(session.model, signals)
        created: list[str] = []
        for d in drafts:
            name = d.get("name", "").strip()
            summary = d.get("summary", "").strip()
            body = d.get("body", "").strip()
            if not (name and body):
                continue
            if self._is_duplicate(user_id, name):
                continue
            self.add(user_id, name, summary, body, "induced")
            created.append(name)
            self.observer.log(session.id, None, "skill_induced", {"name": name})
        return created

    def _gather_signals(self, session: Session) -> str:
        """Cheap, pre-compressed evidence: recent user turns of THIS session
        plus its checkpoint labels and summary."""
        parts: list[str] = []
        summ = self.repo.current_summary(session.id)
        if summ:
            parts.append("SUMMARY: " + summ.content)
        for t in self.repo.user_turns_since(session.id, 0):
            c = t.content if isinstance(t.content, str) else str(t.content)
            parts.append("REQUEST: " + c)
        return "\n".join(parts[-50:])

    def _is_duplicate(self, user_id: str, name: str) -> bool:
        norm = name.strip().lower()
        return any(s.name.strip().lower() == norm for s in self.list(user_id))
