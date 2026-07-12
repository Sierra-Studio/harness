# CLI

`harness/interfaces/cli.py` is the console-script entry point
(`harness = "harness.interfaces.cli:entry"` in `pyproject.toml`) — it's an
**application**, not library-internal code: it's
the one place that opts into `Config.from_env()`, picks a provider from what's
configured (`detect_provider`), and picks a repository from `DATABASE_URL`
(`build_repository`). `Harness` itself never does either of those on its own.

```bash
uv run harness init-db     # create schema in DATABASE_URL
uv run harness chat [user] # interactive session (uses OpenRouter/Azure if configured)
uv run harness serve [host] [port]
                            # HTTP server streaming turns as Server-Sent Events
                            # (host/port also via HARNESS_HTTP_HOST/HARNESS_HTTP_PORT)
uv run harness add-skill <user_id> <name> <summary> [body]
                            # author a skill (body read from stdin if omitted)
uv run harness list-skills <user_id>
```

Without `uv`: `python -m harness.interfaces.cli init-db | chat | serve | ...`.

## `harness chat`

In an interactive terminal, `chat` launches a full-screen
[Textual](https://textual.textualize.io/) app (`harness/interfaces/tui.py`): a
scrollable message history, a docked input, and a status bar with an animated
"thinking…" / "running <tool>…" spinner plus a live token-budget readout. Each
turn runs in a worker thread, so the UI never freezes while the model works.
`Ctrl+L` clears the history; `Ctrl+C` / `Ctrl+D` quits.

When stdout/stdin aren't a TTY (piped, CI, `--plain`), or if Textual can't
import, it falls back to a line-based [Rich](https://rich.readthedocs.io/) REPL
that renders the same widgets inline. Both paths share the renderables in
`harness/interfaces/ui.py`: assistant text as **live markdown**, clean one-line
tool calls/results, and `RenderUI` payloads drawn as real terminal widgets
(cards, tables, bar charts, stats).

Both paths open with a full-width welcome card (logo + user, provider, model,
storage, session). The logo is the bundled `harness/interfaces/assets/logo.png`
rendered as background-stripped half-block art (works in every terminal,
including ones without an inline-image protocol); point `HARNESS_LOGO_PATH` at
another PNG to change it, or set it to an empty string to drop the logo.

The harness is multi-tenant, so every session belongs to a user. The CLI
defaults that id to your OS username; override it with a positional arg
(`harness chat alice`) or `HARNESS_USER=alice`.

It auto-connects MCP servers from `MCP_HTTP_SERVERS` (see
[MCP](../concepts/mcp.md)) **and** any added at runtime with `/mcp http` — those
are saved to `~/.harness/mcp_servers.json` so they reconnect on the next launch
(`/mcp remove <name>` forgets one).

**Editing & keys.** `↑`/`↓` recall previous messages; type `/` for ghost-text
command autocomplete; end a line with `\` to continue on the next line;
`Ctrl+C` (or `Esc`) stops a running turn — and quits when idle; `Ctrl+L` clears.

Type `/help` inside the session for the full list. Everything else you type
is sent to the model as a message.

| Command | Does |
|---|---|
| `/help` | show the command list |
| `/session` | session id, model, context window, token spend |
| `/sessions` | list your recent sessions (subject, when, turns, tokens) |
| `/resume [n\|id]` | resume a past session (most recent if no arg) and reload its history |
| `/retry` | re-run your last message |
| `/copy` | copy the last answer to the clipboard |
| `/save [file]` | save the transcript as markdown |
| `/skills` | list this user's saved skills |
| `/skills add <name> <summary> [body...]` | author a new skill (body defaults to the summary) |
| `/tools` | list active tools, grouped by source: `built-in` first, then each `--direct`-exposed MCP server |
| `/mcp` | list connected MCP servers |
| `/mcp http <url> [name] [--direct]` | connect a remote MCP server (saved for next launch) |
| `/mcp stdio <name> <cmd...> [--direct]` | connect a local stdio MCP server (subprocess) |
| `/mcp remove <name>` | forget a saved MCP server |
| `/persona [text\|clear]` | show / set / reset the persona (saved as the default) |
| `/system-prompt [text\|clear]` | show / set / reset a raw system-prompt override (bypasses persona layering) |
| `/model [name]` | show / change this session's model (saved as the default) |
| `/budget [n\|unlimited]` | show / change this session's token budget (saved as the default) |
| `/theme [name]` | change the color theme — TUI only; `Ctrl+P`'s palette also has a theme picker |
| `/clear` | clear the screen (keeps the session) |
| `/new` | close this session and start a fresh one for the same user |
| `/exit`, `/quit`, `exit`, `quit` | end the session |

`--direct` connects the server with `expose="direct"` (its tools become
first-class, sent to the model directly) instead of the default `"index"`
discovery model — see [MCP](../concepts/mcp.md#exposure-policy-index-vs-direct).

## Preferences (`~/.harness/preferences.json`)

`/persona`, `/system-prompt`, `/model`, `/budget`, and `/theme` all apply
immediately **and** persist to `~/.harness/preferences.json` (via
`harness/interfaces/prefs.py`), so they carry over to your next `harness chat`
— the same pattern as `~/.harness/mcp_servers.json` for MCP servers.

Precedence, highest to lowest: a real environment variable you set
(`HARNESS_MODEL`, `TOKEN_BUDGET_PER_SESSION`, `RESPONSE_RESERVE_TOKENS`) always
wins; then a saved preference; then a value from a checked-in `.env` file;
then the library's own built-in default. This is why setting `/model` in chat
reliably takes effect next launch even if `.env` sets `HARNESS_MODEL` — a
`.env` default is deliberately treated as *lower* priority than something you
typed in the app.

`/persona` and `/system-prompt` are mutually exclusive — setting one clears
the other's saved value, matching `Harness`'s own precedence (`system_prompt`
wins over persona layering when both are non-empty). `set_persona`,
`set_session_model`, and `set_session_budget` on `Harness`
(`harness/core/app.py`) are the underlying runtime-control methods these
commands call — usable directly if you're embedding the library yourself.

Rendering lives in `harness/interfaces/ui.py`, kept separate from the argv
dispatch/wiring in `cli.py`.
