"""Skill induction: every N closed sessions, look for recurring request
patterns and promote them into user-owned skills (deduped by embedding).
Runs off the loop's critical path.
"""
from __future__ import annotations

from .config import Config
from .embeddings import Embedder, cosine
from .models import Session
from .observer import Observer
from .provider import Provider
from .repository import Repository

DUP_THRESHOLD = 0.92


class SkillInducer:
    def __init__(self, repo: Repository, provider: Provider, embedder: Embedder,
                 cfg: Config, observer: Observer):
        self.repo = repo
        self.provider = provider
        self.embedder = embedder
        self.cfg = cfg
        self.observer = observer

    def on_session_closed(self, session: Session) -> list[str]:
        """Trigger induction when the user hits a multiple of the cadence.
        Returns the names of any newly created skills."""
        user_id = session.user_id
        closed = self.repo.count_closed_sessions(user_id)
        if closed == 0 or closed % self.cfg.skill_induction_every_sessions != 0:
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
            emb = self.embedder.embed(f"{name}\n{summary}")
            if self._is_duplicate(user_id, emb):
                continue
            self.repo.add_skill(user_id, name, summary, body, "induced", emb)
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

    def _is_duplicate(self, user_id: str, emb: list[float]) -> bool:
        for s in self.repo.list_skills(user_id):
            if s.embedding and cosine(s.embedding, emb) >= DUP_THRESHOLD:
                return True
        return False
