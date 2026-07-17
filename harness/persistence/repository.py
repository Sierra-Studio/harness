"""Persistence layer.

`Repository` is the abstract contract. Three implementations:
  - InMemoryRepository: dependency-free, used for dev / offline demo / tests.
  - SQLiteRepository: single-file persistence, no server (stdlib sqlite3).
  - PostgresRepository: real persistence (psycopg).

The harness logic depends only on `Repository`, never on a concrete backend.
"""

from __future__ import annotations

import abc
import json
import re
import sqlite3
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

from ..models import Session, SessionSummary, Skill, Summary, ToolSpec, Turn, User


def _uuid() -> str:
    return str(uuid.uuid4())


def _fold(text: str) -> str:
    """Lowercase and strip diacritics so accents never break matching.

    "Reunião" -> "reuniao", "résumé" -> "resume". Language-agnostic: it makes
    keyword search insensitive to accents in any Latin-script language.
    """
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _terms(text: str) -> list[str]:
    """Accent-folded alphanumeric word tokens for keyword search."""
    return re.findall(r"[a-z0-9]+", _fold(text))


def _snippet(content: Any, cap: int = 60) -> str:
    """A one-line preview of a turn's content for the session list subject."""
    text = content if isinstance(content, str) else str(content)
    text = " ".join(text.split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


class Repository(abc.ABC):
    # --- users ---
    @abc.abstractmethod
    def get_or_create_user(self, external_id: str) -> User: ...

    # --- sessions ---
    @abc.abstractmethod
    def create_session(
        self, user_id: str, model: str, context_window: int, token_budget: int
    ) -> Session: ...
    @abc.abstractmethod
    def get_session(self, session_id: str) -> Session: ...
    @abc.abstractmethod
    def find_session(self, session_id: str) -> Session | None:
        """Look up a session by id; None if it doesn't exist (bad id, expired
        store, wrong Harness instance's in-memory repo). Safe to call
        speculatively when resuming, unlike `get_session` which assumes the id
        is already known-valid. Performs NO ownership check — see the
        SECURITY note on `Harness.start_session`, which is the only place
        this should be called from."""
    @abc.abstractmethod
    def add_session_tokens(self, session_id: str, delta: int) -> None: ...
    @abc.abstractmethod
    def set_session_status(self, session_id: str, status: str) -> None: ...
    @abc.abstractmethod
    def close_session(self, session_id: str) -> None: ...
    @abc.abstractmethod
    def count_closed_sessions(self, user_id: str) -> int: ...
    @abc.abstractmethod
    def list_sessions(self, user_id: str, limit: int = 20) -> list[SessionSummary]:
        """Recent sessions for a user, newest first, for the `/sessions` picker.
        Each summary's `subject` is the latest checkpoint label, else a snippet
        of the first user message, else empty."""

    # --- turns ---
    @abc.abstractmethod
    def add_turn(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: Any,
        token_count: int,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> Turn: ...
    @abc.abstractmethod
    def active_turns(self, session_id: str) -> list[Turn]: ...
    @abc.abstractmethod
    def mark_out_of_window(self, turn_ids: list[str]) -> None: ...
    @abc.abstractmethod
    def count_user_turns(self, session_id: str) -> int: ...
    @abc.abstractmethod
    def user_turns_since(self, session_id: str, after_user_turn: int) -> list[Turn]: ...

    # --- summaries ---
    @abc.abstractmethod
    def add_summary(
        self,
        session_id: str,
        parent_id: str | None,
        content: str,
        token_count: int,
        covers_until: int,
    ) -> Summary: ...
    @abc.abstractmethod
    def current_summary(self, session_id: str) -> Summary | None: ...

    # --- checkpoints ---
    @abc.abstractmethod
    def add_checkpoint(
        self, session_id: str, user_id: str, at_user_turn: int, label: str
    ) -> None: ...
    @abc.abstractmethod
    def last_checkpoint_turn(self, session_id: str) -> int: ...

    # --- skills (keyword search, no embeddings) ---
    @abc.abstractmethod
    def add_skill(self, user_id: str, name: str, summary: str, body: str, origin: str) -> Skill: ...
    @abc.abstractmethod
    def list_skills(self, user_id: str) -> list[Skill]: ...
    @abc.abstractmethod
    def search_skills(self, user_id: str, query: str, k: int) -> list[Skill]: ...
    @abc.abstractmethod
    def get_skill(self, user_id: str, name: str) -> Skill | None: ...

    # --- tool index (keyword search, no embeddings) ---
    @abc.abstractmethod
    def upsert_tool(
        self, mcp_server: str, name: str, description: str, input_schema: dict
    ) -> None: ...
    @abc.abstractmethod
    def search_tools(self, query: str, k: int) -> list[ToolSpec]: ...
    @abc.abstractmethod
    def get_tool(self, name: str) -> ToolSpec | None: ...

    # --- observability ---
    @abc.abstractmethod
    def add_step_log(
        self,
        session_id: str | None,
        turn_id: str | None,
        step_type: str,
        detail: dict,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
    ) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------
class InMemoryRepository(Repository):
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.sessions: dict[str, Session] = {}
        self.turns: list[Turn] = []
        self.summaries: list[Summary] = []
        self.checkpoints: list[dict] = []
        self.skills: list[Skill] = []
        self.tools: list[ToolSpec] = []
        self.step_logs: list[dict] = []
        self._started: dict[str, datetime] = {}  # session_id -> creation time

    def get_or_create_user(self, external_id: str) -> User:
        for u in self.users.values():
            if u.external_id == external_id:
                return u
        u = User(id=_uuid(), external_id=external_id)
        self.users[u.id] = u
        return u

    def create_session(self, user_id, model, context_window, token_budget) -> Session:
        s = Session(
            id=_uuid(),
            user_id=user_id,
            model=model,
            context_window=context_window,
            token_budget=token_budget,
        )
        self.sessions[s.id] = s
        self._started[s.id] = datetime.now(timezone.utc)
        return s

    def get_session(self, session_id) -> Session:
        return self.sessions[session_id]

    def find_session(self, session_id) -> Session | None:
        return self.sessions.get(session_id)

    def add_session_tokens(self, session_id, delta) -> None:
        self.sessions[session_id].tokens_spent += delta

    def set_session_status(self, session_id, status) -> None:
        self.sessions[session_id].status = status

    def close_session(self, session_id) -> None:
        self.sessions[session_id].status = "closed"

    def count_closed_sessions(self, user_id) -> int:
        return sum(
            1 for s in self.sessions.values() if s.user_id == user_id and s.status == "closed"
        )

    def list_sessions(self, user_id, limit=20) -> list[SessionSummary]:
        sess = [s for s in self.sessions.values() if s.user_id == user_id]
        sess.sort(key=lambda s: self._started.get(s.id) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        out: list[SessionSummary] = []
        for s in sess[:limit]:
            labels = [c["label"] for c in self.checkpoints if c["session_id"] == s.id]
            subject = labels[-1] if labels else ""
            turns = [t for t in sorted(self.turns, key=lambda t: t.idx) if t.session_id == s.id]
            if not subject:
                first_user = next((t for t in turns if t.role == "user"), None)
                subject = _snippet(first_user.content) if first_user else ""
            out.append(
                SessionSummary(
                    id=s.id,
                    subject=subject,
                    status=s.status,
                    tokens_spent=s.tokens_spent,
                    turns=len(turns),
                    started_at=self._started.get(s.id),
                )
            )
        return out

    def add_turn(
        self, session_id, user_id, role, content, token_count, tokens_in=None, tokens_out=None
    ) -> Turn:
        idx = sum(1 for t in self.turns if t.session_id == session_id)
        t = Turn(
            id=_uuid(),
            session_id=session_id,
            user_id=user_id,
            idx=idx,
            role=role,
            content=content,
            token_count=token_count,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            in_window=True,
        )
        self.turns.append(t)
        return t

    def active_turns(self, session_id) -> list[Turn]:
        return sorted(
            [t for t in self.turns if t.session_id == session_id and t.in_window],
            key=lambda t: t.idx,
        )

    def mark_out_of_window(self, turn_ids) -> None:
        ids = set(turn_ids)
        for t in self.turns:
            if t.id in ids:
                t.in_window = False

    def count_user_turns(self, session_id) -> int:
        return sum(1 for t in self.turns if t.session_id == session_id and t.role == "user")

    def user_turns_since(self, session_id, after_user_turn) -> list[Turn]:
        users = [
            t
            for t in sorted(self.turns, key=lambda t: t.idx)
            if t.session_id == session_id and t.role == "user"
        ]
        return users[after_user_turn:]

    def add_summary(self, session_id, parent_id, content, token_count, covers_until) -> Summary:
        s = Summary(
            id=_uuid(),
            session_id=session_id,
            parent_id=parent_id,
            content=content,
            token_count=token_count,
            covers_until=covers_until,
        )
        self.summaries.append(s)
        return s

    def current_summary(self, session_id) -> Summary | None:
        items = [s for s in self.summaries if s.session_id == session_id]
        return items[-1] if items else None

    def add_checkpoint(self, session_id, user_id, at_user_turn, label) -> None:
        self.checkpoints.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "at_user_turn": at_user_turn,
                "label": label,
            }
        )

    def last_checkpoint_turn(self, session_id) -> int:
        items = [c["at_user_turn"] for c in self.checkpoints if c["session_id"] == session_id]
        return max(items) if items else 0

    def add_skill(self, user_id, name, summary, body, origin) -> Skill:
        sk = Skill(
            id=_uuid(), user_id=user_id, name=name, summary=summary, body=body, origin=origin
        )
        self.skills.append(sk)
        return sk

    def list_skills(self, user_id) -> list[Skill]:
        return [s for s in self.skills if s.user_id == user_id]

    def search_skills(self, user_id, query, k) -> list[Skill]:
        items = self.list_skills(user_id)
        terms = set(_terms(query))
        if not terms:
            return items[:k]
        scored: list[tuple[int, Skill]] = []
        for s in items:
            hay = f"{s.name} {s.summary} {s.body}".lower()
            score = sum(1 for term in terms if term in hay)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def get_skill(self, user_id, name) -> Skill | None:
        for s in self.skills:
            if s.user_id == user_id and s.name == name:
                return s
        return None

    def upsert_tool(self, mcp_server, name, description, input_schema) -> None:
        for t in self.tools:
            if t.mcp_server == mcp_server and t.name == name:
                t.description, t.input_schema = description, input_schema
                return
        self.tools.append(
            ToolSpec(
                id=_uuid(),
                mcp_server=mcp_server,
                name=name,
                description=description,
                input_schema=input_schema,
            )
        )

    def search_tools(self, query, k) -> list[ToolSpec]:
        terms = set(_terms(query))
        scored: list[tuple[int, ToolSpec]] = []
        for t in self.tools:
            if not t.enabled:
                continue
            hay = _fold(f"{t.name} {t.description}")
            score = sum(1 for term in terms if term in hay)
            if score or not terms:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def get_tool(self, name) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def add_step_log(
        self,
        session_id,
        turn_id,
        step_type,
        detail,
        tokens_in=None,
        tokens_out=None,
        latency_ms=None,
    ) -> None:
        self.step_logs.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "step_type": step_type,
                "detail": detail,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
            }
        )


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------
class PostgresRepository(Repository):
    """psycopg3. Skills and the tool index are searched via Postgres full-text."""

    def __init__(self, dsn: str) -> None:
        import psycopg  # local import; optional dependency

        self._psycopg = psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)

    # -- helpers --
    def _row(self, sql: str, params: tuple = ()):  # one row
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def _rows(self, sql: str, params: tuple = ()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _exec(self, sql: str, params: tuple = ()):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    # -- users --
    def get_or_create_user(self, external_id) -> User:
        self._exec(
            "INSERT INTO users(external_id) VALUES(%s) ON CONFLICT (external_id) DO NOTHING",
            (external_id,),
        )
        r = self._row("SELECT id, external_id FROM users WHERE external_id=%s", (external_id,))
        return User(id=str(r[0]), external_id=r[1])

    # -- sessions --
    def create_session(self, user_id, model, context_window, token_budget) -> Session:
        r = self._row(
            """INSERT INTO sessions(user_id, model, context_window, token_budget)
               VALUES(%s,%s,%s,%s) RETURNING id""",
            (user_id, model, context_window, token_budget),
        )
        return Session(
            id=str(r[0]),
            user_id=user_id,
            model=model,
            context_window=context_window,
            token_budget=token_budget,
        )

    def get_session(self, session_id) -> Session:
        r = self._row(
            """SELECT id,user_id,model,context_window,token_budget,tokens_spent,status
               FROM sessions WHERE id=%s""",
            (session_id,),
        )
        return Session(
            id=str(r[0]),
            user_id=str(r[1]),
            model=r[2],
            context_window=r[3],
            token_budget=r[4],
            tokens_spent=r[5],
            status=r[6],
        )

    def find_session(self, session_id) -> Session | None:
        try:
            r = self._row(
                """SELECT id,user_id,model,context_window,token_budget,tokens_spent,status
                   FROM sessions WHERE id=%s""",
                (session_id,),
            )
        except self._psycopg.DataError:
            # A malformed id (not a uuid at all) is the same "bad id" case the
            # contract promises None for — not an exception. InMemoryRepository
            # already behaves this way; keep both backends identical.
            return None
        if not r:
            return None
        return Session(
            id=str(r[0]),
            user_id=str(r[1]),
            model=r[2],
            context_window=r[3],
            token_budget=r[4],
            tokens_spent=r[5],
            status=r[6],
        )

    def add_session_tokens(self, session_id, delta) -> None:
        self._exec(
            "UPDATE sessions SET tokens_spent=tokens_spent+%s WHERE id=%s", (delta, session_id)
        )

    def set_session_status(self, session_id, status) -> None:
        self._exec("UPDATE sessions SET status=%s WHERE id=%s", (status, session_id))

    def close_session(self, session_id) -> None:
        self._exec("UPDATE sessions SET status='closed', ended_at=now() WHERE id=%s", (session_id,))

    def count_closed_sessions(self, user_id) -> int:
        return self._row(
            "SELECT count(*) FROM sessions WHERE user_id=%s AND status='closed'", (user_id,)
        )[0]

    def list_sessions(self, user_id, limit=20) -> list[SessionSummary]:
        rows = self._rows(
            """
            SELECT s.id, s.status, s.tokens_spent, s.started_at,
                   (SELECT count(*) FROM turns t WHERE t.session_id = s.id) AS turns,
                   COALESCE(
                     (SELECT c.label FROM checkpoints c WHERE c.session_id = s.id
                        ORDER BY c.at_user_turn DESC LIMIT 1),
                     (SELECT left(t.content #>> '{}', 80) FROM turns t
                        WHERE t.session_id = s.id AND t.role = 'user'
                        ORDER BY t.idx LIMIT 1),
                     ''
                   ) AS subject
            FROM sessions s
            WHERE s.user_id = %s
            ORDER BY s.started_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return [
            SessionSummary(
                id=str(r[0]),
                subject=_snippet(r[5] or "", 60),
                status=r[1],
                tokens_spent=r[2],
                turns=r[4],
                started_at=r[3],
            )
            for r in rows
        ]

    # -- turns --
    def add_turn(
        self, session_id, user_id, role, content, token_count, tokens_in=None, tokens_out=None
    ) -> Turn:
        import json

        idx = self._row("SELECT count(*) FROM turns WHERE session_id=%s", (session_id,))[0]
        r = self._row(
            """INSERT INTO turns(session_id,user_id,idx,role,content,token_count,tokens_in,tokens_out)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (
                session_id,
                user_id,
                idx,
                role,
                json.dumps(content),
                token_count,
                tokens_in,
                tokens_out,
            ),
        )
        return Turn(
            id=str(r[0]),
            session_id=session_id,
            user_id=user_id,
            idx=idx,
            role=role,
            content=content,
            token_count=token_count,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def active_turns(self, session_id) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count,tokens_in,tokens_out
               FROM turns WHERE session_id=%s AND in_window ORDER BY idx""",
            (session_id,),
        )
        return [
            Turn(
                id=str(r[0]),
                session_id=session_id,
                user_id=str(r[1]),
                idx=r[2],
                role=r[3],
                content=r[4],
                token_count=r[5],
                tokens_in=r[6],
                tokens_out=r[7],
            )
            for r in rows
        ]

    def mark_out_of_window(self, turn_ids) -> None:
        if not turn_ids:
            return
        self._exec("UPDATE turns SET in_window=false WHERE id = ANY(%s)", (list(turn_ids),))

    def count_user_turns(self, session_id) -> int:
        return self._row(
            "SELECT count(*) FROM turns WHERE session_id=%s AND role='user'", (session_id,)
        )[0]

    def user_turns_since(self, session_id, after_user_turn) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count FROM turns
               WHERE session_id=%s AND role='user' ORDER BY idx OFFSET %s""",
            (session_id, after_user_turn),
        )
        return [
            Turn(
                id=str(r[0]),
                session_id=session_id,
                user_id=str(r[1]),
                idx=r[2],
                role=r[3],
                content=r[4],
                token_count=r[5],
            )
            for r in rows
        ]

    # -- summaries --
    def add_summary(self, session_id, parent_id, content, token_count, covers_until) -> Summary:
        r = self._row(
            """INSERT INTO summaries(session_id,parent_id,content,token_count,covers_until)
               VALUES(%s,%s,%s,%s,%s) RETURNING id""",
            (session_id, parent_id, content, token_count, covers_until),
        )
        return Summary(
            id=str(r[0]),
            session_id=session_id,
            parent_id=parent_id,
            content=content,
            token_count=token_count,
            covers_until=covers_until,
        )

    def current_summary(self, session_id) -> Summary | None:
        r = self._row(
            """SELECT id,parent_id,content,token_count,covers_until FROM summaries
               WHERE session_id=%s ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        )
        if not r:
            return None
        return Summary(
            id=str(r[0]),
            session_id=session_id,
            parent_id=str(r[1]) if r[1] else None,
            content=r[2],
            token_count=r[3],
            covers_until=r[4],
        )

    # -- checkpoints --
    def add_checkpoint(self, session_id, user_id, at_user_turn, label) -> None:
        self._exec(
            "INSERT INTO checkpoints(session_id,user_id,at_user_turn,label) VALUES(%s,%s,%s,%s)",
            (session_id, user_id, at_user_turn, label),
        )

    def last_checkpoint_turn(self, session_id) -> int:
        r = self._row(
            "SELECT max(at_user_turn) FROM checkpoints WHERE session_id=%s", (session_id,)
        )
        return r[0] or 0

    # -- skills (Postgres full-text search, no embeddings) --
    def add_skill(self, user_id, name, summary, body, origin) -> Skill:
        r = self._row(
            """INSERT INTO skills(user_id,name,summary,body,origin)
               VALUES(%s,%s,%s,%s,%s) RETURNING id""",
            (user_id, name, summary, body, origin),
        )
        return Skill(
            id=str(r[0]), user_id=user_id, name=name, summary=summary, body=body, origin=origin
        )

    def list_skills(self, user_id) -> list[Skill]:
        rows = self._rows(
            "SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s", (user_id,)
        )
        return [
            Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2], body=r[3], origin=r[4])
            for r in rows
        ]

    def search_skills(self, user_id, query, k) -> list[Skill]:
        terms = _terms(query)
        if not terms:
            rows = self._rows(
                "SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s LIMIT %s",
                (user_id, k),
            )
        else:
            tsquery = " | ".join(terms)  # OR of terms for recall
            rows = self._rows(
                """SELECT id,name,summary,body,origin,
                          ts_rank(to_tsvector('english', name||' '||summary||' '||body),
                                  to_tsquery('english', %s)) AS rank
                   FROM skills
                   WHERE user_id=%s
                     AND to_tsvector('english', name||' '||summary||' '||body)
                         @@ to_tsquery('english', %s)
                   ORDER BY rank DESC LIMIT %s""",
                (tsquery, user_id, tsquery, k),
            )
        return [
            Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2], body=r[3], origin=r[4])
            for r in rows
        ]

    def get_skill(self, user_id, name) -> Skill | None:
        r = self._row(
            """SELECT id,name,summary,body,origin FROM skills
               WHERE user_id=%s AND name=%s LIMIT 1""",
            (user_id, name),
        )
        if not r:
            return None
        return Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2], body=r[3], origin=r[4])

    # -- tool index (Postgres full-text search, no embeddings) --
    def upsert_tool(self, mcp_server, name, description, input_schema) -> None:
        import json

        self._exec(
            """INSERT INTO tool_index(mcp_server,name,description,input_schema)
               VALUES(%s,%s,%s,%s)
               ON CONFLICT (mcp_server,name) DO UPDATE
                 SET description=EXCLUDED.description,
                     input_schema=EXCLUDED.input_schema""",
            (mcp_server, name, description, json.dumps(input_schema)),
        )

    # Accent-folded, language-agnostic document expression (matches the GIN
    # index in schema.sql). 'simple' avoids English-only stemming; f_unaccent
    # makes both sides accent-insensitive.
    _DOC = "to_tsvector('simple', f_unaccent(name||' '||coalesce(description,'')))"
    _COLS = "id,mcp_server,name,description,input_schema"

    def search_tools(self, query, k) -> list[ToolSpec]:
        terms = _terms(query)  # already accent-folded + lowercased
        if not terms:
            rows = self._rows(f"SELECT {self._COLS} FROM tool_index WHERE enabled LIMIT %s", (k,))
            return [self._toolspec(r) for r in rows]

        # 1) Full-text with prefix matching (`:*`) so partial/variant terms hit.
        tsquery = " | ".join(f"{t}:*" for t in terms)  # OR of prefixes for recall
        rows = self._rows(
            f"""SELECT {self._COLS},
                       ts_rank({self._DOC}, to_tsquery('simple', %s)) AS rank
                FROM tool_index
                WHERE enabled AND {self._DOC} @@ to_tsquery('simple', %s)
                ORDER BY rank DESC LIMIT %s""",
            (tsquery, tsquery, k),
        )
        if rows:
            return [self._toolspec(r) for r in rows]

        # 2) Substring fallback (accent-insensitive) for anything FTS missed,
        #    e.g. a term that is a fragment of a tool name with no word boundary.
        clauses = " OR ".join(
            ["f_unaccent(name) ILIKE %s OR f_unaccent(description) ILIKE %s"] * len(terms)
        )
        params: list[Any] = []
        for t in terms:
            params += [f"%{t}%", f"%{t}%"]
        params.append(k)
        rows = self._rows(
            f"SELECT {self._COLS} FROM tool_index WHERE enabled AND ({clauses}) LIMIT %s",
            tuple(params),
        )
        return [self._toolspec(r) for r in rows]

    def get_tool(self, name) -> ToolSpec | None:
        r = self._row(f"SELECT {self._COLS} FROM tool_index WHERE name=%s LIMIT 1", (name,))
        return self._toolspec(r) if r else None

    @staticmethod
    def _toolspec(r) -> ToolSpec:
        return ToolSpec(
            id=str(r[0]), mcp_server=r[1], name=r[2], description=r[3], input_schema=r[4]
        )

    # -- observability --
    def add_step_log(
        self,
        session_id,
        turn_id,
        step_type,
        detail,
        tokens_in=None,
        tokens_out=None,
        latency_ms=None,
    ) -> None:
        import json

        self._exec(
            """INSERT INTO step_logs(session_id,turn_id,step_type,detail,tokens_in,tokens_out,latency_ms)
               VALUES(%s,%s,%s,%s,%s,%s,%s)""",
            (session_id, turn_id, step_type, json.dumps(detail), tokens_in, tokens_out, latency_ms),
        )


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------
class SQLiteRepository(Repository):
    """Single-file persistence backed by the stdlib `sqlite3` module.

    A zero-dependency middle ground between InMemory (lost on exit) and
    Postgres (needs a server). SQLite has no `gen_random_uuid()` and no
    `to_tsvector` full-text, so ids are generated in Python and skill/tool
    keyword search is ranked in Python with the same accent-folded `_terms`
    matching that InMemoryRepository uses. JSON columns (`content`,
    `input_schema`, `detail`) and timestamps are stored as TEXT.
    """

    # Embedded DDL — schema.sql is Postgres-specific (extensions, jsonb, GIN),
    # so SQLite gets its own portable version. `idx INTEGER` etc. map cleanly.
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
      id          TEXT PRIMARY KEY,
      external_id TEXT UNIQUE NOT NULL,
      created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
      id             TEXT PRIMARY KEY,
      user_id        TEXT NOT NULL REFERENCES users(id),
      model          TEXT NOT NULL,
      context_window INTEGER NOT NULL,
      token_budget   INTEGER NOT NULL,
      tokens_spent   INTEGER NOT NULL DEFAULT 0,
      status         TEXT NOT NULL DEFAULT 'open',
      started_at     TEXT NOT NULL,
      ended_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
    CREATE TABLE IF NOT EXISTS turns (
      id          TEXT PRIMARY KEY,
      session_id  TEXT NOT NULL REFERENCES sessions(id),
      user_id     TEXT NOT NULL REFERENCES users(id),
      idx         INTEGER NOT NULL,
      role        TEXT NOT NULL,
      content     TEXT NOT NULL,
      token_count INTEGER NOT NULL,
      tokens_in   INTEGER,
      tokens_out  INTEGER,
      in_window   INTEGER NOT NULL DEFAULT 1,
      created_at  TEXT NOT NULL,
      UNIQUE (session_id, idx)
    );
    CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, idx);
    CREATE TABLE IF NOT EXISTS summaries (
      id           TEXT PRIMARY KEY,
      session_id   TEXT NOT NULL REFERENCES sessions(id),
      parent_id    TEXT REFERENCES summaries(id),
      content      TEXT NOT NULL,
      token_count  INTEGER NOT NULL,
      covers_until INTEGER NOT NULL,
      created_at   TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, created_at);
    CREATE TABLE IF NOT EXISTS checkpoints (
      id           TEXT PRIMARY KEY,
      session_id   TEXT NOT NULL REFERENCES sessions(id),
      user_id      TEXT NOT NULL REFERENCES users(id),
      at_user_turn INTEGER NOT NULL,
      label        TEXT NOT NULL,
      created_at   TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS skills (
      id         TEXT PRIMARY KEY,
      user_id    TEXT NOT NULL REFERENCES users(id),
      name       TEXT NOT NULL,
      summary    TEXT NOT NULL,
      body       TEXT NOT NULL,
      origin     TEXT NOT NULL DEFAULT 'induced',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_skills_user ON skills(user_id);
    CREATE TABLE IF NOT EXISTS tool_index (
      id           TEXT PRIMARY KEY,
      mcp_server   TEXT NOT NULL,
      name         TEXT NOT NULL,
      description  TEXT NOT NULL,
      input_schema TEXT NOT NULL,
      enabled      INTEGER NOT NULL DEFAULT 1,
      UNIQUE (mcp_server, name)
    );
    CREATE TABLE IF NOT EXISTS step_logs (
      id         TEXT PRIMARY KEY,
      session_id TEXT REFERENCES sessions(id),
      turn_id    TEXT REFERENCES turns(id),
      step_type  TEXT NOT NULL,
      detail     TEXT,
      tokens_in  INTEGER,
      tokens_out INTEGER,
      latency_ms INTEGER,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_steplogs_session ON step_logs(session_id, created_at);
    """

    def __init__(self, path: str) -> None:
        # check_same_thread=False + a lock: the harness may touch the repo from
        # more than one thread, and sqlite3 forbids cross-thread use otherwise.
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self.conn.executescript(self._SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    # -- helpers (thread-safe; '?' placeholders) --
    def _row(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchone()

    def _rows(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchall()

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            self.conn.execute(sql, params)

    # -- users --
    def get_or_create_user(self, external_id) -> User:
        r = self._row("SELECT id, external_id FROM users WHERE external_id=?", (external_id,))
        if r:
            return User(id=r[0], external_id=r[1])
        uid = _uuid()
        self._exec(
            "INSERT INTO users(id, external_id, created_at) VALUES(?,?,?)",
            (uid, external_id, self._now()),
        )
        return User(id=uid, external_id=external_id)

    # -- sessions --
    def create_session(self, user_id, model, context_window, token_budget) -> Session:
        sid = _uuid()
        self._exec(
            """INSERT INTO sessions(id,user_id,model,context_window,token_budget,started_at)
               VALUES(?,?,?,?,?,?)""",
            (sid, user_id, model, context_window, token_budget, self._now()),
        )
        return Session(
            id=sid,
            user_id=user_id,
            model=model,
            context_window=context_window,
            token_budget=token_budget,
        )

    def _session(self, r) -> Session:
        return Session(
            id=r[0],
            user_id=r[1],
            model=r[2],
            context_window=r[3],
            token_budget=r[4],
            tokens_spent=r[5],
            status=r[6],
        )

    def get_session(self, session_id) -> Session:
        r = self._row(
            """SELECT id,user_id,model,context_window,token_budget,tokens_spent,status
               FROM sessions WHERE id=?""",
            (session_id,),
        )
        return self._session(r)

    def find_session(self, session_id) -> Session | None:
        r = self._row(
            """SELECT id,user_id,model,context_window,token_budget,tokens_spent,status
               FROM sessions WHERE id=?""",
            (session_id,),
        )
        return self._session(r) if r else None

    def add_session_tokens(self, session_id, delta) -> None:
        self._exec(
            "UPDATE sessions SET tokens_spent=tokens_spent+? WHERE id=?", (delta, session_id)
        )

    def set_session_status(self, session_id, status) -> None:
        self._exec("UPDATE sessions SET status=? WHERE id=?", (status, session_id))

    def close_session(self, session_id) -> None:
        self._exec(
            "UPDATE sessions SET status='closed', ended_at=? WHERE id=?",
            (self._now(), session_id),
        )

    def count_closed_sessions(self, user_id) -> int:
        return self._row(
            "SELECT count(*) FROM sessions WHERE user_id=? AND status='closed'", (user_id,)
        )[0]

    def list_sessions(self, user_id, limit=20) -> list[SessionSummary]:
        rows = self._rows(
            """
            SELECT s.id, s.status, s.tokens_spent, s.started_at,
                   (SELECT count(*) FROM turns t WHERE t.session_id = s.id) AS turns,
                   COALESCE(
                     (SELECT c.label FROM checkpoints c WHERE c.session_id = s.id
                        ORDER BY c.at_user_turn DESC LIMIT 1),
                     (SELECT t.content FROM turns t
                        WHERE t.session_id = s.id AND t.role = 'user'
                        ORDER BY t.idx LIMIT 1),
                     ''
                   ) AS subject
            FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        out: list[SessionSummary] = []
        for r in rows:
            subject = r[5]
            # The first-user-message fallback comes back as a JSON-encoded
            # `content`; decode it to a display string. Checkpoint labels are
            # plain text and pass through _snippet unchanged.
            if subject and subject not in ("",):
                try:
                    decoded = json.loads(subject)
                    subject = decoded if isinstance(decoded, str) else subject
                except (json.JSONDecodeError, TypeError):
                    pass
            out.append(
                SessionSummary(
                    id=r[0],
                    subject=_snippet(subject or "", 60),
                    status=r[1],
                    tokens_spent=r[2],
                    turns=r[4],
                    started_at=self._dt(r[3]),
                )
            )
        return out

    # -- turns --
    def add_turn(
        self, session_id, user_id, role, content, token_count, tokens_in=None, tokens_out=None
    ) -> Turn:
        idx = self._row("SELECT count(*) FROM turns WHERE session_id=?", (session_id,))[0]
        tid = _uuid()
        self._exec(
            """INSERT INTO turns(id,session_id,user_id,idx,role,content,token_count,
                                 tokens_in,tokens_out,in_window,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,1,?)""",
            (
                tid,
                session_id,
                user_id,
                idx,
                role,
                json.dumps(content),
                token_count,
                tokens_in,
                tokens_out,
                self._now(),
            ),
        )
        return Turn(
            id=tid,
            session_id=session_id,
            user_id=user_id,
            idx=idx,
            role=role,
            content=content,
            token_count=token_count,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            in_window=True,
        )

    def _turn(self, r, session_id) -> Turn:
        return Turn(
            id=r[0],
            session_id=session_id,
            user_id=r[1],
            idx=r[2],
            role=r[3],
            content=json.loads(r[4]),
            token_count=r[5],
            tokens_in=r[6],
            tokens_out=r[7],
            in_window=bool(r[8]),
        )

    def active_turns(self, session_id) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count,tokens_in,tokens_out,in_window
               FROM turns WHERE session_id=? AND in_window=1 ORDER BY idx""",
            (session_id,),
        )
        return [self._turn(r, session_id) for r in rows]

    def mark_out_of_window(self, turn_ids) -> None:
        ids = list(turn_ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._exec(
            f"UPDATE turns SET in_window=0 WHERE id IN ({placeholders})", tuple(ids)
        )

    def count_user_turns(self, session_id) -> int:
        return self._row(
            "SELECT count(*) FROM turns WHERE session_id=? AND role='user'", (session_id,)
        )[0]

    def user_turns_since(self, session_id, after_user_turn) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count,tokens_in,tokens_out,in_window
               FROM turns WHERE session_id=? AND role='user'
               ORDER BY idx LIMIT -1 OFFSET ?""",
            (session_id, after_user_turn),
        )
        return [self._turn(r, session_id) for r in rows]

    # -- summaries --
    def add_summary(self, session_id, parent_id, content, token_count, covers_until) -> Summary:
        sid = _uuid()
        self._exec(
            """INSERT INTO summaries(id,session_id,parent_id,content,token_count,covers_until,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (sid, session_id, parent_id, content, token_count, covers_until, self._now()),
        )
        return Summary(
            id=sid,
            session_id=session_id,
            parent_id=parent_id,
            content=content,
            token_count=token_count,
            covers_until=covers_until,
        )

    def current_summary(self, session_id) -> Summary | None:
        r = self._row(
            """SELECT id,parent_id,content,token_count,covers_until FROM summaries
               WHERE session_id=? ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        )
        if not r:
            return None
        return Summary(
            id=r[0],
            session_id=session_id,
            parent_id=r[1],
            content=r[2],
            token_count=r[3],
            covers_until=r[4],
        )

    # -- checkpoints --
    def add_checkpoint(self, session_id, user_id, at_user_turn, label) -> None:
        self._exec(
            """INSERT INTO checkpoints(id,session_id,user_id,at_user_turn,label,created_at)
               VALUES(?,?,?,?,?,?)""",
            (_uuid(), session_id, user_id, at_user_turn, label, self._now()),
        )

    def last_checkpoint_turn(self, session_id) -> int:
        r = self._row(
            "SELECT max(at_user_turn) FROM checkpoints WHERE session_id=?", (session_id,)
        )
        return (r[0] if r else None) or 0

    # -- skills (keyword search ranked in Python, mirrors InMemory) --
    def add_skill(self, user_id, name, summary, body, origin) -> Skill:
        sid = _uuid()
        self._exec(
            """INSERT INTO skills(id,user_id,name,summary,body,origin,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (sid, user_id, name, summary, body, origin, self._now()),
        )
        return Skill(id=sid, user_id=user_id, name=name, summary=summary, body=body, origin=origin)

    def list_skills(self, user_id) -> list[Skill]:
        rows = self._rows(
            "SELECT id,name,summary,body,origin FROM skills WHERE user_id=?", (user_id,)
        )
        return [
            Skill(id=r[0], user_id=user_id, name=r[1], summary=r[2], body=r[3], origin=r[4])
            for r in rows
        ]

    def search_skills(self, user_id, query, k) -> list[Skill]:
        items = self.list_skills(user_id)
        terms = set(_terms(query))
        if not terms:
            return items[:k]
        scored: list[tuple[int, Skill]] = []
        for s in items:
            hay = _fold(f"{s.name} {s.summary} {s.body}")
            score = sum(1 for term in terms if term in hay)
            if score:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def get_skill(self, user_id, name) -> Skill | None:
        r = self._row(
            "SELECT id,name,summary,body,origin FROM skills WHERE user_id=? AND name=? LIMIT 1",
            (user_id, name),
        )
        if not r:
            return None
        return Skill(id=r[0], user_id=user_id, name=r[1], summary=r[2], body=r[3], origin=r[4])

    # -- tool index (keyword search ranked in Python, mirrors InMemory) --
    def upsert_tool(self, mcp_server, name, description, input_schema) -> None:
        self._exec(
            """INSERT INTO tool_index(id,mcp_server,name,description,input_schema)
               VALUES(?,?,?,?,?)
               ON CONFLICT(mcp_server,name) DO UPDATE
                 SET description=excluded.description,
                     input_schema=excluded.input_schema""",
            (_uuid(), mcp_server, name, description, json.dumps(input_schema)),
        )

    def _toolspec(self, r) -> ToolSpec:
        return ToolSpec(
            id=r[0],
            mcp_server=r[1],
            name=r[2],
            description=r[3],
            input_schema=json.loads(r[4]),
            enabled=bool(r[5]),
        )

    def search_tools(self, query, k) -> list[ToolSpec]:
        rows = self._rows(
            "SELECT id,mcp_server,name,description,input_schema,enabled FROM tool_index WHERE enabled=1"
        )
        tools = [self._toolspec(r) for r in rows]
        terms = set(_terms(query))
        scored: list[tuple[int, ToolSpec]] = []
        for t in tools:
            hay = _fold(f"{t.name} {t.description}")
            score = sum(1 for term in terms if term in hay)
            if score or not terms:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def get_tool(self, name) -> ToolSpec | None:
        r = self._row(
            "SELECT id,mcp_server,name,description,input_schema,enabled FROM tool_index WHERE name=? LIMIT 1",
            (name,),
        )
        return self._toolspec(r) if r else None

    # -- observability --
    def add_step_log(
        self,
        session_id,
        turn_id,
        step_type,
        detail,
        tokens_in=None,
        tokens_out=None,
        latency_ms=None,
    ) -> None:
        self._exec(
            """INSERT INTO step_logs(id,session_id,turn_id,step_type,detail,
                                     tokens_in,tokens_out,latency_ms,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                _uuid(),
                session_id,
                turn_id,
                step_type,
                json.dumps(detail),
                tokens_in,
                tokens_out,
                latency_ms,
                self._now(),
            ),
        )


def build_repository(cfg) -> Repository:
    """Factory: SQLite for `sqlite://…` URLs, Postgres for any other
    DATABASE_URL, else in-memory."""
    url = cfg.database_url
    if url:
        if url.startswith("sqlite://"):
            # sqlite:///abs/path.db -> "/abs/path.db"; sqlite://:memory: -> ":memory:";
            # sqlite:// -> ":memory:". (Follows the SQLAlchemy 3-slash convention.)
            path = url[len("sqlite://"):]
            return SQLiteRepository(path or ":memory:")
        return PostgresRepository(url)
    return InMemoryRepository()
