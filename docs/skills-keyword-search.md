# Skills: drop vector search, fetch bodies, mirror the tools pattern

Reference for an implementation pass on `harness/`. Not yet applied.

## Problem

1. **Bodies are unreachable.** `_get_skills` (`harness/tools.py:114`) returns
   `[{name, summary}]` in *every* path. There is no tool to fetch a skill's
   `body`, so the model can discover a skill but never read its procedure.
   The comment at `tools.py:123` ("body loaded only if asked") describes an
   intent that was never implemented.

2. **Vector search isn't earning its keep.** Skill recall uses embeddings +
   cosine similarity, but:
   - The local embedder (`embeddings.py:48`) is a SHA1 hash-bag-of-words
     fallback — non-semantic, worse than keyword overlap — and the remote path
     silently falls back to it on any error (`embeddings.py:28`).
   - Skills are low-cardinality (tens per user), so a linear keyword scan is
     plenty.
   - Tools already use keyword/full-text search; skills should match.

## Target design

Make skills identical in shape to tools: **keyword search for discovery + a
by-name fetch for the body.**

- `SearchTools(query)` → `[{name, description}]`, then `GetTool(name)` → full spec.
- `GetSkills(query)` → `[{name, summary}]`, then `GetSkill(name)` → full body.

## Change set (5 areas)

### 1. `repository.py` — keyword `search_skills` (both backends)
- In-memory (`:204`): replace the cosine sort with the `_terms` overlap scorer
  used by `search_tools` (`:217`), matching over `name + summary + body`.
- Postgres (`:399`): replace `ORDER BY embedding <=> %s` with the
  `to_tsvector` / `ts_rank` full-text query (mirror `search_tools` at `:407`).
- Drop the `embedding` argument from the signature and the protocol stub (`:84`).

### 2. `repository.py` — `add_skill`
- In-memory (`:195`) and Postgres (`:385`): drop the `embedding` param; remove
  it from the INSERT (`:387`).

### 3. `tools.py` — split discovery from retrieval
- Add `GetSkill` to `BUILTINS` (`:17`), register it (`:44`), dispatch it (`:72`).
- `_get_skills(query)` (`:114`): keyword search, return `[{name, summary}]`;
  remove the `self.embedder.embed(...)` call (`:117`).
- New `_get_skill(name)`: return the full `body` (mirror `_get_tool` at `:106`).

### 4. `skills.py` — dedup without cosine
- Replace embed + cosine dedup (`:43`, `:63`, `DUP_THRESHOLD` at `:14`) with
  normalized-name equality (or token-overlap / Jaccard on `name + summary`).
- Drop the `Embedder` import/field and the `cosine` import.

### 5. Cleanup
- `models.py:59`: remove the `embedding` field from `Skill`.
- `schema.sql`: drop the `embedding` column from `skills`; update the comment
  at `:78`.
- `app.py:36`: stop passing `embedder` to `SkillInducer`.
- `prompt.py:41,46`: teach the two-step flow (GetSkills → GetSkill).
- `embeddings.py` + `config.py:61` (`EMBEDDING_*`) + `app.py:27`: once nothing
  else references `Embedder`, delete the module and config. **Verify no other
  consumer first** (tool index is already keyword-based, so this should be the
  last use).

## Tradeoff to acknowledge

With a *real* remote embedder, vector search catches paraphrases (query
"change the server passwords" matching a skill summarized "rotate prod
credentials"); keyword search will not. Mitigate by indexing the skill **body**
into the full-text haystack so more vocabulary is searchable. Given the
non-semantic local fallback and low skill counts, this is an acceptable trade.

## Net result

Skills and tools share one mechanism (keyword search + by-name fetch), the
embedding pipeline drops out entirely, and the model can finally read and follow
a skill's procedure.
