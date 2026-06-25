"""Persistence layer.

`Repository` is the abstract contract. Two implementations:
  - InMemoryRepository: dependency-free, used for dev / offline demo / tests.
  - PostgresRepository: real persistence (psycopg).

The harness logic depends only on `Repository`, never on a concrete backend.
"""
from __future__ import annotations

import abc
import re
import unicodedata
import uuid
from typing import Any, Optional

from .models import Session, Skill, Summary, ToolSpec, Turn, User


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

    # --- skills (keyword search, no embeddings) ---
    @abc.abstractmethod
    def add_skill(self, user_id: str, name: str, summary: str, body: str,
                  origin: str) -> Skill: ...
    @abc.abstractmethod
    def list_skills(self, user_id: str) -> list[Skill]: ...
    @abc.abstractmethod
    def search_skills(self, user_id: str, query: str, k: int) -> list[Skill]: ...
    @abc.abstractmethod
    def get_skill(self, user_id: str, name: str) -> Optional[Skill]: ...

    # --- tool index (keyword search, no embeddings) ---
    @abc.abstractmethod
    def upsert_tool(self, mcp_server: str, name: str, description: str,
                    input_schema: dict) -> None: ...
    @abc.abstractmethod
    def search_tools(self, query: str, k: int) -> list[ToolSpec]: ...
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

    def add_skill(self, user_id, name, summary, body, origin) -> Skill:
        sk = Skill(id=_uuid(), user_id=user_id, name=name, summary=summary,
                   body=body, origin=origin)
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

    def get_skill(self, user_id, name) -> Optional[Skill]:
        for s in self.skills:
            if s.user_id == user_id and s.name == name:
                return s
        return None

    def upsert_tool(self, mcp_server, name, description, input_schema) -> None:
        for t in self.tools:
            if t.mcp_server == mcp_server and t.name == name:
                t.description, t.input_schema = description, input_schema
                return
        self.tools.append(ToolSpec(id=_uuid(), mcp_server=mcp_server, name=name,
                                   description=description, input_schema=input_schema))

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

    # -- skills (Postgres full-text search, no embeddings) --
    def add_skill(self, user_id, name, summary, body, origin) -> Skill:
        r = self._row(
            """INSERT INTO skills(user_id,name,summary,body,origin)
               VALUES(%s,%s,%s,%s,%s) RETURNING id""",
            (user_id, name, summary, body, origin))
        return Skill(id=str(r[0]), user_id=user_id, name=name, summary=summary,
                     body=body, origin=origin)

    def list_skills(self, user_id) -> list[Skill]:
        rows = self._rows(
            "SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s", (user_id,))
        return [Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2],
                      body=r[3], origin=r[4]) for r in rows]

    def search_skills(self, user_id, query, k) -> list[Skill]:
        terms = _terms(query)
        if not terms:
            rows = self._rows(
                "SELECT id,name,summary,body,origin FROM skills WHERE user_id=%s LIMIT %s",
                (user_id, k))
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
                (tsquery, user_id, tsquery, k))
        return [Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2],
                      body=r[3], origin=r[4]) for r in rows]

    def get_skill(self, user_id, name) -> Optional[Skill]:
        r = self._row(
            """SELECT id,name,summary,body,origin FROM skills
               WHERE user_id=%s AND name=%s LIMIT 1""", (user_id, name))
        if not r:
            return None
        return Skill(id=str(r[0]), user_id=user_id, name=r[1], summary=r[2],
                     body=r[3], origin=r[4])

    # -- tool index (Postgres full-text search, no embeddings) --
    def upsert_tool(self, mcp_server, name, description, input_schema) -> None:
        import json
        self._exec(
            """INSERT INTO tool_index(mcp_server,name,description,input_schema)
               VALUES(%s,%s,%s,%s)
               ON CONFLICT (mcp_server,name) DO UPDATE
                 SET description=EXCLUDED.description,
                     input_schema=EXCLUDED.input_schema""",
            (mcp_server, name, description, json.dumps(input_schema)))

    # Accent-folded, language-agnostic document expression (matches the GIN
    # index in schema.sql). 'simple' avoids English-only stemming; f_unaccent
    # makes both sides accent-insensitive.
    _DOC = ("to_tsvector('simple', "
            "f_unaccent(name||' '||coalesce(description,'')))")
    _COLS = "id,mcp_server,name,description,input_schema"

    def search_tools(self, query, k) -> list[ToolSpec]:
        terms = _terms(query)  # already accent-folded + lowercased
        if not terms:
            rows = self._rows(
                f"SELECT {self._COLS} FROM tool_index WHERE enabled LIMIT %s", (k,))
            return [self._toolspec(r) for r in rows]

        # 1) Full-text with prefix matching (`:*`) so partial/variant terms hit.
        tsquery = " | ".join(f"{t}:*" for t in terms)  # OR of prefixes for recall
        rows = self._rows(
            f"""SELECT {self._COLS},
                       ts_rank({self._DOC}, to_tsquery('simple', %s)) AS rank
                FROM tool_index
                WHERE enabled AND {self._DOC} @@ to_tsquery('simple', %s)
                ORDER BY rank DESC LIMIT %s""",
            (tsquery, tsquery, k))
        if rows:
            return [self._toolspec(r) for r in rows]

        # 2) Substring fallback (accent-insensitive) for anything FTS missed,
        #    e.g. a term that is a fragment of a tool name with no word boundary.
        clauses = " OR ".join(
            ["f_unaccent(name) ILIKE %s OR f_unaccent(description) ILIKE %s"]
            * len(terms))
        params: list[Any] = []
        for t in terms:
            params += [f"%{t}%", f"%{t}%"]
        params.append(k)
        rows = self._rows(
            f"SELECT {self._COLS} FROM tool_index WHERE enabled AND ({clauses}) "
            "LIMIT %s", tuple(params))
        return [self._toolspec(r) for r in rows]

    def get_tool(self, name) -> Optional[ToolSpec]:
        r = self._row(
            f"SELECT {self._COLS} FROM tool_index WHERE name=%s LIMIT 1", (name,))
        return self._toolspec(r) if r else None

    @staticmethod
    def _toolspec(r) -> ToolSpec:
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
