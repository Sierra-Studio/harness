"""Unit tests for the CLI/TUI rendering helpers in harness.interfaces.ui.

These cover the bug-prone pure functions (result summarization, argument
formatting, RenderUI node building) — the parts that turned raw JSON tool
output into readable one-liners and a UI tree into terminal widgets. No
Textual/threads here, so they stay deterministic."""

from __future__ import annotations

import json
import os

from rich.console import Console

from harness.interfaces import ui


def _render(renderable) -> str:
    """Render to plain text so we can assert on visible content."""
    with open(os.devnull, "w") as devnull:
        con = Console(width=80, file=devnull, record=True)
        con.print(renderable)
        return con.export_text()


def test_tool_display_name_unwraps_calltool():
    assert ui.tool_display_name("CallTool", {"name": "search_meetings"}) == "search_meetings"
    assert ui.tool_display_name("Bash", {"command": "ls"}) == "Bash"


def test_tool_args_str_is_compact_and_unwraps_calltool():
    s = ui.tool_args_str("CallTool", {"name": "search_meetings", "arguments": {"from_date": "2026-05-11"}})
    assert 'from_date="2026-05-11"' in s
    assert "arguments" not in s  # unwrapped, not shown as a nested key
    assert "name" not in s


def test_summarize_result_collapses_escaped_newlines():
    raw = json.dumps([{"name": "search_meetings", "description": "Search.\n\nMore text."}])
    out = ui.summarize_result(raw)
    assert "\\n" not in out  # escaped newlines gone
    assert out.startswith("1 result: search_meetings")


def test_summarize_result_extracts_mcp_text():
    raw = json.dumps({"content": [{"type": "text", "text": "15 meetings found"}], "isError": False})
    assert ui.summarize_result(raw) == "15 meetings found"


def test_is_error_result():
    assert ui.is_error_result("ERROR: bad payload") is True
    assert ui.is_error_result(json.dumps({"content": [], "isError": True})) is True
    assert ui.is_error_result(json.dumps({"content": [], "isError": False})) is False
    assert ui.is_error_result("just text") is False


def test_tools_renderable_groups_by_mcp_server():
    class Builtin:
        def __init__(self, name, guidance):
            self.name, self.guidance = name, guidance

    class Client:
        def __init__(self, name):
            self.name = name

    class Proxy:
        def __init__(self, name, guidance, client_name):
            self.name, self.guidance = name, guidance
            self._client = Client(client_name)

    tools = [
        Builtin("Bash", "- Bash(command): universal fallback."),
        Proxy("run_python", "- run_python: execute a snippet.", "sandbox"),
        Proxy("search_meetings", "- search_meetings: search meetings.", "fellow"),
    ]
    out = _render(ui.tools_renderable(tools))
    assert "built-in" in out and "sandbox" in out and "fellow" in out
    # built-in group must render before the MCP server groups
    assert out.index("built-in") < out.index("fellow")
    assert out.index("built-in") < out.index("sandbox")
    assert "run_python" in out and "search_meetings" in out


def test_ui_renderable_rejects_invalid_root():
    assert ui.ui_renderable(None) is None
    assert ui.ui_renderable({"no": "type"}) is None
    assert ui.ui_renderable("string") is None


def test_ui_renderable_renders_widget_tree():
    root = {
        "type": "Stack",
        "children": [
            {"type": "Heading", "text": "Meetings"},
            {"type": "Table", "columns": ["Name", "Date"], "rows": [["Sync", "Jul 9"]]},
            {"type": "Chart", "series": [{"label": "wk1", "value": 3}, {"label": "wk2", "value": 5}]},
            {"type": "Callout", "tone": "success", "text": "done"},
        ],
    }
    out = _render(ui.ui_renderable(root))
    assert "Meetings" in out
    assert "Sync" in out and "Jul 9" in out
    assert "wk1" in out and "█" in out  # bar chart drew bars
    assert "done" in out


def test_list_sessions_and_resolve_resume():
    from harness.persistence import InMemoryRepository

    repo = InMemoryRepository()
    uid = repo.get_or_create_user("u").id
    s1 = repo.create_session(uid, "m", 1000, 0)
    repo.add_turn(s1.id, uid, "user", "first thing", 3)
    s2 = repo.create_session(uid, "m", 1000, 0)
    repo.add_turn(s2.id, uid, "user", "second thing", 3)

    rows = repo.list_sessions(uid)
    assert [r.id for r in rows] == [s2.id, s1.id]  # newest first
    assert rows[0].subject == "second thing" and rows[0].turns == 1

    assert ui.resolve_resume(repo, uid, "").id == s2.id  # default = most recent
    assert ui.resolve_resume(repo, uid, "2").id == s1.id  # by list index
    assert ui.resolve_resume(repo, uid, s1.id[:8]).id == s1.id  # by id prefix
    assert ui.resolve_resume(repo, uid, "zzzz") is None


def test_transcript_markdown():
    from harness.models import Turn

    turns = [
        Turn(id="1", session_id="s", user_id="u", idx=0, role="user", content="hi", token_count=1),
        Turn(id="2", session_id="s", user_id="u", idx=1, role="assistant", content="hello", token_count=1),
        Turn(id="3", session_id="s", user_id="u", idx=2, role="tool", content="ignored", token_count=1),
    ]
    md = ui.transcript_markdown(turns)
    assert "## user" in md and "hi" in md
    assert "## assistant" in md and "hello" in md
    assert "ignored" not in md  # tool turns are omitted


def test_input_suggester_completes_slash_commands_only(tmp_path):
    import asyncio

    from harness.interfaces.tui import InputSuggester

    sug = InputSuggester(["/help", "/skills", "/skills add"])

    async def ask(v):
        return await sug.get_suggestion(v)

    assert asyncio.run(ask("/sk")) == "/skills"
    assert asyncio.run(ask("/skills a")) == "/skills add"
    assert asyncio.run(ask("@app")) is None  # @ handled by the dropdown, not ghost text
    assert asyncio.run(ask("plain")) is None


def test_list_path_matches_and_mention_partial(tmp_path):
    from harness.interfaces.tui import _list_path_matches, _mention_partial

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / ".git").mkdir()

    # directories sort first and get a trailing slash; ignore-list is skipped
    assert _list_path_matches(tmp_path, "") == ["app/", "README.md"]
    assert _list_path_matches(tmp_path, "RE") == ["README.md"]
    assert _list_path_matches(tmp_path, "app/") == ["app/main.py"]
    assert _list_path_matches(tmp_path, ".g") == []  # .git ignored

    assert _mention_partial("look @app/ma") == "app/ma"
    assert _mention_partial("@") == ""
    assert _mention_partial("no mention here") is None


def test_expand_mentions_inlines_files(tmp_path):
    from harness.interfaces.tui import HarnessApp

    (tmp_path / "a.txt").write_text("AAA")
    (tmp_path / "b.txt").write_text("BBB")

    class Stub:
        _root = tmp_path

    sent, attached = HarnessApp._expand_mentions(Stub(), "see @a.txt and @b.txt, plus @missing.txt")
    assert attached == ["a.txt", "b.txt"]  # missing file skipped
    assert '<file path="a.txt">\nAAA\n</file>' in sent
    assert '<file path="b.txt">\nBBB\n</file>' in sent
    assert "@missing.txt" in sent  # unresolved mention stays as text
    assert HarnessApp._expand_mentions(Stub(), "no mentions") == ("no mentions", [])


def test_expand_mentions_expands_directories(tmp_path):
    from harness.interfaces.tui import HarnessApp

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "one.py").write_text("1")
    (tmp_path / "pkg" / "two.py").write_text("2")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "x.pyc").write_text("bin")

    class Stub:
        _root = tmp_path

    # @pkg/ (or @pkg) pulls in every file under the dir, pruning ignore-listed dirs
    sent, attached = HarnessApp._expand_mentions(Stub(), "review @pkg/")
    assert set(attached) == {"pkg/one.py", "pkg/two.py"}
    assert '<file path="pkg/one.py">' in sent and '<file path="pkg/two.py">' in sent
    assert "__pycache__" not in sent


def test_keyboard_scrolls_log_without_stealing_input_focus():
    import asyncio
    import dataclasses

    from rich.text import Text
    from textual.containers import VerticalScroll

    from harness.core import Harness
    from harness.interfaces.tui import HarnessApp
    from harness.llm.provider import FakeProvider
    from harness.settings import Config

    async def scenario():
        h = Harness(
            dataclasses.replace(Config(), database_url=""),
            system_prompt="s",
            provider=FakeProvider(context_window=8000),
        )
        app = HarnessApp(h, h.start_session("u"), "u", "F", "f", [])
        async with app.run_test() as pilot:
            await pilot.pause()
            for i in range(80):
                app._write(Text(f"line {i}"))
            await pilot.pause()
            log = app.query_one("#log", VerticalScroll)
            bottom = log.scroll_offset.y
            assert bottom > 0
            await pilot.press("pageup")
            await pilot.pause()
            assert log.scroll_offset.y < bottom  # scrolled up
            await pilot.press("ctrl+home")
            await pilot.pause()
            assert log.scroll_offset.y == 0  # jumped to top
            await pilot.press("ctrl+end")
            await pilot.pause()
            assert log.scroll_offset.y == bottom  # back to bottom
            await pilot.press("h", "i")  # input kept focus
            await pilot.pause()
            assert app.query_one("#prompt").value == "hi"

    asyncio.run(scenario())


def test_at_dropdown_multi_match_navigate_and_pick(tmp_path):
    import asyncio
    import dataclasses

    from harness.core import Harness
    from harness.interfaces.tui import HarnessApp, PromptInput
    from harness.llm.provider import FakeProvider
    from harness.settings import Config
    from harness.tools import LocalSubprocessSandbox

    (tmp_path / "alpha.txt").write_text("A")
    (tmp_path / "apple.txt").write_text("APPLE")

    async def scenario():
        sb = LocalSubprocessSandbox(workspace=str(tmp_path))
        h = Harness(
            dataclasses.replace(Config(), database_url=""),
            system_prompt="s",
            provider=FakeProvider(context_window=8000),
            sandbox=sb,
        )
        app = HarnessApp(h, h.start_session("u"), "u", "F", "f", [])
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#ac").display is False
            await pilot.press("@", "a")
            await pilot.pause()
            assert app._ac_open and app._ac_matches == ["alpha.txt", "apple.txt"]
            await pilot.press("down")  # move selection to apple.txt
            await pilot.pause()
            assert app._ac_sel == 1
            await pilot.press("enter")  # pick it (does not submit)
            await pilot.pause()
            assert app.query_one("#prompt", PromptInput).value == "@apple.txt "
            assert app._ac_open is False and app._busy is False

    asyncio.run(scenario())


def test_cli_version_flag(capsys):
    from harness import __version__
    from harness.interfaces.cli import main

    for flag in ("--version", "-V", "version"):
        assert main(["harness", flag]) == 0
        assert capsys.readouterr().out.strip() == f"harness {__version__}"


def test_cli_help_flag_exits_zero_but_unknown_is_error(capsys):
    from harness.interfaces.cli import main

    assert main(["harness", "--help"]) == 0
    capsys.readouterr()
    assert main(["harness", "bogus-cmd"]) == 1  # unknown command still errors


def test_turn_summary_flags_over_budget():
    class R:
        status = "ok"
        steps = 7
        tokens_spent = 604_730

    out = _render(ui.turn_summary_renderable(R(), 500_000))
    assert "over budget" in out
    assert "604,730 tokens" in out
