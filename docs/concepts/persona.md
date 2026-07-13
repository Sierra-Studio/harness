# Persona

`harness/memory/persona.py` assembles the system prompt in ordered layers so the stable
identity sits first (cache-friendly) and per-call guidance follows:

```
[ persona / identity ]   <- PERSONA.md if present & non-empty, else DEFAULT_IDENTITY
[ tool guidance       ]  <- composed from the ACTIVE tools' own guidance snippets
[ extra message       ]  <- optional caller-supplied system_message
```

The tool-guidance layer is not hardcoded: each `Tool` carries its own
`guidance` snippet, and `compose_tool_guidance` assembles only the ones for
tools that are actually active — the prompt never mentions a tool that isn't
present.

## Setting a persona

```python
Harness(persona="You are Atlas, a terse senior SRE. You think in shell commands.")
```

Or via a file:

- Create `PERSONA.md` in the working directory, or
- Point `HARNESS_PERSONA_PATH` (env, via `MemoryConfig.persona_path`) at one.

`PERSONA.md` is loaded fresh on each prompt build — delete it (or empty its
content) to fall back to the built-in default identity. Comment-only files
(HTML comments) count as empty.

To bypass the layered assembly entirely, pass `system_prompt=...` directly to
`Harness`.

## Skills in the prompt

`skills_block()` renders up to `MemoryConfig.skills_in_prompt_limit` (default
30) of the user's skills directly into the prompt. Above that limit, the model
falls back to the `SearchSkills` tool for the long tail — the hybrid keeps
prompt cost bounded regardless of how many skills a user has accumulated.
