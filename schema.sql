-- Harness schema (Postgres).
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS unaccent;  -- accent-insensitive tool search

-- IMMUTABLE wrapper around unaccent() (the bare function is only STABLE) so it
-- can back an expression index. Two-arg form pins the dictionary explicitly.
CREATE OR REPLACE FUNCTION f_unaccent(text) RETURNS text
  LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT AS
$$ SELECT public.unaccent('public.unaccent', $1) $$;

CREATE TABLE IF NOT EXISTS users (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id text UNIQUE NOT NULL,
  created_at  timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id        uuid NOT NULL REFERENCES users(id),
  model          text NOT NULL,
  context_window int  NOT NULL,
  token_budget   int  NOT NULL,
  tokens_spent   int  NOT NULL DEFAULT 0,
  status         text NOT NULL DEFAULT 'open',   -- open | closed | budget_exhausted
  started_at     timestamptz DEFAULT now(),
  ended_at       timestamptz
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- Each TURN is one message; central unit of memory.
CREATE TABLE IF NOT EXISTS turns (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id  uuid NOT NULL REFERENCES sessions(id),
  user_id     uuid NOT NULL REFERENCES users(id),
  idx         int  NOT NULL,                  -- order within the session
  role        text NOT NULL,                  -- user | system | assistant | tool
  content     jsonb NOT NULL,
  token_count int  NOT NULL,
  tokens_in   int,                            -- prompt tokens (model-generated turns)
  tokens_out  int,                            -- completion tokens
  in_window   boolean NOT NULL DEFAULT true,  -- still in the active window?
  created_at  timestamptz DEFAULT now(),
  UNIQUE (session_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, idx);

-- Chained summaries: each one folds the previous + the evicted turns.
CREATE TABLE IF NOT EXISTS summaries (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   uuid NOT NULL REFERENCES sessions(id),
  parent_id    uuid REFERENCES summaries(id),
  content      text NOT NULL,
  token_count  int  NOT NULL,
  covers_until int  NOT NULL,                 -- max turn.idx covered
  created_at   timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, created_at);

-- Checkpoint: subject classification every N user turns.
CREATE TABLE IF NOT EXISTS checkpoints (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   uuid NOT NULL REFERENCES sessions(id),
  user_id      uuid NOT NULL REFERENCES users(id),
  at_user_turn int  NOT NULL,
  label        text NOT NULL,
  created_at   timestamptz DEFAULT now()
);

-- Skills owned by a user (authored or induced).
-- Searched by keyword (Postgres full-text), no embeddings.
CREATE TABLE IF NOT EXISTS skills (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    uuid NOT NULL REFERENCES users(id),
  name       text NOT NULL,
  summary    text NOT NULL,
  body       text NOT NULL,
  origin     text NOT NULL DEFAULT 'induced',  -- authored | induced
  created_at timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_skills_user ON skills(user_id);
-- GIN index backing the full-text SearchSkills query.
CREATE INDEX IF NOT EXISTS idx_skills_fts ON skills
  USING gin (to_tsvector('english', name || ' ' || summary || ' ' || body));

-- Index of Index Tools (from MCP). NOT injected into the system prompt.
-- Searched by keyword (Postgres full-text), no embeddings.
CREATE TABLE IF NOT EXISTS tool_index (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  mcp_server   text NOT NULL,
  name         text NOT NULL,
  description  text NOT NULL,
  input_schema jsonb NOT NULL,
  enabled      boolean NOT NULL DEFAULT true,
  UNIQUE (mcp_server, name)
);
-- GIN index backing SearchTools: language-agnostic ('simple', no English-only
-- stemming) over accent-folded text, so queries match regardless of accents or
-- input language. Dropped first so re-applying the schema picks up the new
-- definition even if an older index of the same name exists.
DROP INDEX IF EXISTS idx_tool_index_fts;
CREATE INDEX IF NOT EXISTS idx_tool_index_fts ON tool_index
  USING gin (to_tsvector('simple', f_unaccent(name || ' ' || coalesce(description, ''))));

-- Observability: one row per loop step.
CREATE TABLE IF NOT EXISTS step_logs (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid REFERENCES sessions(id),
  turn_id    uuid REFERENCES turns(id),
  step_type  text NOT NULL,                    -- model_call | tool_call | summarize | checkpoint | skill_induced
  detail     jsonb,
  tokens_in  int,
  tokens_out int,
  latency_ms int,
  created_at timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_steplogs_session ON step_logs(session_id, created_at);
