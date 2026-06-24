"""Persistence layer.

`Repository` is the abstract contract. Two implementations:
  - InMemoryRepository: dependency-free, used for dev / offline demo / tests.
  - PostgresRepository: real persistence (psycopg + pgvector).

The harness logic depends only on `Repository`, never on a concrete backend.
"""
from __future__ import annotations

import abc
import uuid
from typing import Any, Optional

from .embeddings import cosine
from .models import Session, Skill, Summary, ToolSpec, Turn, User


def _uuid() -> str:
    return str(uuid.uuid4())


class Repository(abc.ABC):
    # --- users ---
    @abc.abstractmethod
    def get_or_create_user(self, external_id: str) -> User: ...

    # --- sessions ---
    @abc.abstractmethod
    def create_session(self, user_id: str, model: str, context_window: int,
                       token_budget: int) -> Session: ...
    @abc.abstractmethod
    def get_session(self, session_id: str) -> Session: ...
    @abc.abstractmethod
    def add_session_tokens(self, session_id: str, delta: int) -> None: ...
    @abc.abstractmethod
    def set_session_status(self, session_id: str, status: str) -> None: ...
    @abc.abstractmethod
    def close_session(self, session_id: str) -> None: ...
    @abc.abstractmethod
    def count_closed_sessions(self, user_id: str) -> int: ...

    # --- turns ---
    @abc.abstractmethod
    def add_turn(self, session_id: str, user_id: str, role: str, content: Any,
                 token_count: int, tokens_in: Optional[int] = None,
                 tokens_out: Optional[int] = None) -> Turn: ...
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
    def add_summary(self, session_id: str, parent_id: Optional[str], content: str,
                    token_count: int, covers_until: int) -> Summary: ...
    @abc.abstractmethod
    def current_summary(self, session_id: str) -> Optional[Summary]: ...

    # --- checkpoints ---
    @abc.abstractmethod
    def add_checkpoint(self, session_id: str, user_id: str, at_user_turn: int,
                       label: str) -> None: ...
    @abc.abstractmethod
    def last_checkpoint_turn(self, session_id: str) -> int: ...

    # --- skills ---
    @abc.abstractmethod
    def add_skill(self, user_id: str, name: str, summary: str, body: str,
                  origin: str, embedding: list[float]) -> Skill: ...
    @abc.abstractmethod
    def list_skills(self, user_id: str) -> list[Skill]: ...
    @abc.abstractmethod
    def search_skills(self, user_id: str, embedding: list[float], k: int) -> list[Skill]: ...

    # --- tool index ---
    @abc.abstractmethod
    def upsert_tool(self, mcp_server: str, name: str, description: str,
                    input_schema: dict, embedding: list[float]) -> None: ...
    @abc.abstractmethod
    def search_tools(self, embedding: list[float], k: int) -> list[ToolSpec]: ...
    @abc.abstractmethod
    def get_tool(self, name: str) -> Optional[ToolSpec]: ...

    # --- observability ---
    @abc.abstractmethod
    def add_step_log(self, session_id: Optional[str], turn_id: Optional[str],
                     step_type: str, detail: dict, tokens_in: Optional[int] = None,
                     tokens_out: Optional[int] = None,
                     latency_ms: Optional[int] = None) -> None: ...


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

    def get_or_create_user(self, external_id: str) -> User:
        for u in self.users.values():
            if u.external_id == external_id:
                return u
        u = User(id=_uuid(), external_id=external_id)
        self.users[u.id] = u
        return u

    def create_session(self, user_id, model, context_window, token_budget) -> Session:
        s = Session(id=_uuid(), user_id=user_id, model=model,
                    context_window=context_window, token_budget=token_budget)
        self.sessions[s.id] = s
        return s

    def get_session(self, session_id) -> Session:
        return self.sessions[session_id]

    def add_session_tokens(self, session_id, delta) -> None:
        self.sessions[session_id].tokens_spent += delta

    def set_session_status(self, session_id, status) -> None:
        self.sessions[session_id].status = status

    def close_session(self, session_id) -> None:
        self.sessions[session_id].status = "closed"

    def count_closed_sessions(self, user_id) -> int:
        return sum(1 for s in self.sessions.values()
                   if s.user_id == user_id and s.status == "closed")

    def add_turn(self, session_id, user_id, role, content, token_count,
                 tokens_in=None, tokens_out=None) -> Turn:
        idx = sum(1 for t in self.turns if t.session_id == session_id)
        t = Turn(id=_uuid(), session_id=session_id, user_id=user_id, idx=idx,
                 role=role, content=content, token_count=token_count,
                 tokens_in=tokens_in, tokens_out=tokens_out, in_window=True)
        self.turns.append(t)
        return t

    def active_turns(self, session_id) -> list[Turn]:
        return sorted([t for t in self.turns
                       if t.session_id == session_id and t.in_window],
                      key=lambda t: t.idx)

    def mark_out_of_window(self, turn_ids) -> None:
        ids = set(turn_ids)
        for t in self.turns:
            if t.id in ids:
                t.in_window = False

    def count_user_turns(self, session_id) -> int:
        return sum(1 for t in self.turns
                   if t.session_id == session_id and t.role == "user")

    def user_turns_since(self, session_id, after_user_turn) -> list[Turn]:
        users = [t for t in sorted(self.turns, key=lambda t: t.idx)
                 if t.session_id == session_id and t.role == "user"]
        return users[after_user_turn:]

    def add_summary(self, session_id, parent_id, content, token_count, covers_until) -> Summary:
        s = Summary(id=_uuid(), session_id=session_id, parent_id=parent_id,
                    content=content, token_count=token_count, covers_until=covers_until)
        self.summaries.append(s)
        return s

    def current_summary(self, session_id) -> Optional[Summary]:
        items = [s for s in self.summaries if s.session_id == session_id]
        return items[-1] if items else None

    def add_checkpoint(self, session_id, user_id, at_user_turn, label) -> None:
        self.checkpoints.append({"session_id": session_id, "user_id": user_id,
                                 "at_user_turn": at_user_turn, "label": label})

    def last_checkpoint_turn(self, session_id) -> int:
        items = [c["at_user_turn"] for c in self.checkpoints
                 if c["session_id"] == session_id]
        return max(items) if items else 0

    def add_skill(self, user_id, name, summary, body, origin, embedding) -> Skill:
        sk = Skill(id=_uuid(), user_id=user_id, name=name, summary=summary,
                   body=body, origin=origin, embedding=embedding)
        self.skills.append(sk)
        return sk

    def list_skills(self, user_id) -> list[Skill]:
        return [s for s in self.skills if s.user_id == user_id]

    def search_skills(self, user_id, embedding, k) -> list[Skill]:
        items = self.list_skills(user_id)
        items.sort(key=lambda s: cosine(s.embedding, embedding), reverse=True)
        return items[:k]

    def upsert_tool(self, mcp_server, name, description, input_schema, embedding) -> None:
        for t in self.tools:
            if t.mcp_server == mcp_server and t.name == name:
                t.description, t.input_schema, t.embedding = description, input_schema, embedding
                return
        self.tools.append(ToolSpec(id=_uuid(), mcp_server=mcp_server, name=name,
                                   description=description, input_schema=input_schema,
                                   embedding=embedding))

    def search_tools(self, embedding, k) -> list[ToolSpec]:
        items = [t for t in self.tools if t.enabled]
        items.sort(key=lambda t: cosine(t.embedding, embedding), reverse=True)
        return items[:k]

    def get_tool(self, name) -> Optional[ToolSpec]:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def add_step_log(self, session_id, turn_id, step_type, detail,
                     tokens_in=None, tokens_out=None, latency_ms=None) -> None:
        self.step_logs.append({"session_id": session_id, "turn_id": turn_id,
                               "step_type": step_type, "detail": detail,
                               "tokens_in": tokens_in, "tokens_out": tokens_out,
                               "latency_ms": latency_ms})


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------
class PostgresRepository(Repository):
    """psycopg3 + pgvector. Embeddings are stored as pgvector literals."""

    def __init__(self, dsn: str) -> None:
        import psycopg  # local import; optional dependency

        self._psycopg = psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)

    # -- helpers --
    @staticmethod
    def _vec(embedding: list[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in embedding) + "]"

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
            (external_id,))
        r = self._row("SELECT id, external_id FROM users WHERE external_id=%s", (external_id,))
        return User(id=str(r[0]), external_id=r[1])

    # -- sessions --
    def create_session(self, user_id, model, context_window, token_budget) -> Session:
        r = self._row(
            """INSERT INTO sessions(user_id, model, context_window, token_budget)
               VALUES(%s,%s,%s,%s) RETURNING id""",
            (user_id, model, context_window, token_budget))
        return Session(id=str(r[0]), user_id=user_id, model=model,
                       context_window=context_window, token_budget=token_budget)

    def get_session(self, session_id) -> Session:
        r = self._row(
            """SELECT id,user_id,model,context_window,token_budget,tokens_spent,status
               FROM sessions WHERE id=%s""", (session_id,))
        return Session(id=str(r[0]), user_id=str(r[1]), model=r[2], context_window=r[3],
                       token_budget=r[4], tokens_spent=r[5], status=r[6])

    def add_session_tokens(self, session_id, delta) -> None:
        self._exec("UPDATE sessions SET tokens_spent=tokens_spent+%s WHERE id=%s",
                   (delta, session_id))

    def set_session_status(self, session_id, status) -> None:
        self._exec("UPDATE sessions SET status=%s WHERE id=%s", (status, session_id))

    def close_session(self, session_id) -> None:
        self._exec("UPDATE sessions SET status='closed', ended_at=now() WHERE id=%s",
                   (session_id,))

    def count_closed_sessions(self, user_id) -> int:
        return self._row("SELECT count(*) FROM sessions WHERE user_id=%s AND status='closed'",
                         (user_id,))[0]

    # -- turns --
    def add_turn(self, session_id, user_id, role, content, token_count,
                 tokens_in=None, tokens_out=None) -> Turn:
        import json
        idx = self._row("SELECT count(*) FROM turns WHERE session_id=%s", (session_id,))[0]
        r = self._row(
            """INSERT INTO turns(session_id,user_id,idx,role,content,token_count,tokens_in,tokens_out)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (session_id, user_id, idx, role, json.dumps(content), token_count,
             tokens_in, tokens_out))
        return Turn(id=str(r[0]), session_id=session_id, user_id=user_id, idx=idx,
                    role=role, content=content, token_count=token_count,
                    tokens_in=tokens_in, tokens_out=tokens_out)

    def active_turns(self, session_id) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count,tokens_in,tokens_out
               FROM turns WHERE session_id=%s AND in_window ORDER BY idx""", (session_id,))
        return [Turn(id=str(r[0]), session_id=session_id, user_id=str(r[1]), idx=r[2],
                     role=r[3], content=r[4], token_count=r[5], tokens_in=r[6],
                     tokens_out=r[7]) for r in rows]

    def mark_out_of_window(self, turn_ids) -> None:
        if not turn_ids:
            return
        self._exec("UPDATE turns SET in_window=false WHERE id = ANY(%s)", (list(turn_ids),))

    def count_user_turns(self, session_id) -> int:
        return self._row(
            "SELECT count(*) FROM turns WHERE session_id=%s AND role='user'",
            (session_id,))[0]

    def user_turns_since(self, session_id, after_user_turn) -> list[Turn]:
        rows = self._rows(
            """SELECT id,user_id,idx,role,content,token_count FROM turns
               WHERE session_id=%s AND role='user' ORDER BY idx OFFSET %s""",
            (session_id, after_user_turn))
        return [Turn(id=str(r[0]), session_id=session_id, user_id=str(r[1]), idx=r[2],
                     role=r[3], content=r[4], token_count=r[5]) for r in rows]

    # -- summaries --
    def add_summary(self, session_id, parent_id, content, token_count, covers_until) -> Summary:
        r = self._row(
            """INSERT INTO summaries(session_id,parent_id,content,token_count,covers_until)
               VALUES(%s,%s,%s,%s,%s) RETURNING id""",
            (session_id, parent_id, content, token_count, covers_until))
        return Summary(id=str(r[0]), session_id=session_id, parent_id=parent_id,
                       content=content, token_count=token_count, covers_until=covers_until)

    def current_summary(self, session_id) -> Optional[Summary]:
        r = self._row(
            """SELECT id,parent_id,content,token_count,covers_until FROM summaries
               WHERE session_id=%s ORDER BY created_at DESC LIMIT 1""", (session_id,))
        if not r:
            return None
        return Summary(id=str(r[0]), session_id=session_id,
                       parent_id=str(r[1]) if r[1] else None, content=r[2],
                       token_count=r[3], covers_until=r[4])

    # -- checkpoints --
    def add_checkpoint(self, session_id, user_id, at_user_turn, label) -> None:
        self._exec(
            "INSERT INTO checkpoints(session_id,user_id,at_user_turn,label) VALUES(%s,%s,%s,%s)",
            (session_id, user_id, at_user_turn, label))

    def last_checkpoint_turn(self, session_id) -> int:
        r = self._row(
            "SELECT max(at_user_turn) FROM checkpoints WHERE session_id=%s", (session_id,))
        return r[0] or 0

    # -- skills --
    def add_skill(self, user_id, name, summary, body, origin, embedding) -> Skill:
        r = self._row(
            """INSERT INTO skills(user_id,name,summary,body,origin,embedding)
               VALUES(%s,%s,%s,%s,%s,%s) RETURNING id""",
            (user_id, name, summary, body, origin, self._vec(embedding)))
        return Skill(id=str(r[0]), user_id=user_id, name=name, summary=summary,
                     body=body, origin=origin, embedding=embedding)

    def list_skills(self, user_id) -> list[Skill]:
        rows = self._rows(
            "SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s", (user_id,))
        return [Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2],
                      body=r[3], origin=r[4]) for r in rows]

    def search_skills(self, user_id, embedding, k) -> list[Skill]:
        rows = self._rows(
            """SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s
               ORDER BY embedding <=> %s LIMIT %s""",
            (user_id, self._vec(embedding), k))
        return [Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2],
                      body=r[3], origin=r[4]) for r in rows]

    # -- tool index --
    def upsert_tool(self, mcp_server, name, description, input_schema, embedding) -> None:
        import json
        self._exec(
            """INSERT INTO tool_index(mcp_server,name,description,input_schema,embedding)
               VALUES(%s,%s,%s,%s,%s)
               ON CONFLICT (mcp_server,name) DO UPDATE
                 SET description=EXCLUDED.description,
                     input_schema=EXCLUDED.input_schema,
                     embedding=EXCLUDED.embedding""",
            (mcp_server, name, description, json.dumps(input_schema), self._vec(embedding)))

    def search_tools(self, embedding, k) -> list[ToolSpec]:
        rows = self._rows(
            """SELECT id,mcp_server,name,description,input_schema FROM tool_index
               WHERE enabled ORDER BY embedding <=> %s LIMIT %s""",
            (self._vec(embedding), k))
        return [ToolSpec(id=str(r[0]), mcp_server=r[1], name=r[2], description=r[3],
                         input_schema=r[4]) for r in rows]

    def get_tool(self, name) -> Optional[ToolSpec]:
        r = self._row(
            """SELECT id,mcp_server,name,description,input_schema FROM tool_index
               WHERE name=%s LIMIT 1""", (name,))
        if not r:
            return None
        return ToolSpec(id=str(r[0]), mcp_server=r[1], name=r[2], description=r[3],
                        input_schema=r[4])

    # -- observability --
    def add_step_log(self, session_id, turn_id, step_type, detail,
                     tokens_in=None, tokens_out=None, latency_ms=None) -> None:
        import json
        self._exec(
            """INSERT INTO step_logs(session_id,turn_id,step_type,detail,tokens_in,tokens_out,latency_ms)
               VALUES(%s,%s,%s,%s,%s,%s,%s)""",
            (session_id, turn_id, step_type, json.dumps(detail), tokens_in,
             tokens_out, latency_ms))


def build_repository(cfg) -> Repository:
    """Factory: Postgres when DATABASE_URL is set, else in-memory."""
    if cfg.database_url:
        return PostgresRepository(cfg.database_url)
    return InMemoryRepository()
