"""Execution backend for the Bash tool.

`SandboxBackend` is the pluggable contract. `LocalSubprocessSandbox` is the
local implementation: one working directory per session, command timeout, output
capped with head/tail elision, and — unlike a naive subprocess — the working
directory PERSISTS across calls within a session (a `cd` in one call carries to
the next). It is NOT isolated — swap it for a kernel-isolated backend
(gVisor / Firecracker / K8s Pod) for real multi-tenant safety. See the companion
architecture guide.
"""

from __future__ import annotations

import abc
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# marker used to read back the working directory after a command runs
_CWD_MARK = "__HARNESS_CWD__:"


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    cwd: str = ""


class SandboxBackend(abc.ABC):
    @abc.abstractmethod
    def exec(self, session_id: str, command: str, timeout: int = 60) -> ExecResult: ...

    @abc.abstractmethod
    def destroy(self, session_id: str) -> None: ...

    # File I/O for the Write/Edit tools. Relative paths resolve against the same
    # persisted working directory `exec` uses, so all three tools see one view of
    # the filesystem. Backends that can't expose a filesystem leave these as-is
    # (the tools surface the NotImplementedError message to the model).
    def read_file(self, session_id: str, path: str) -> str:
        raise NotImplementedError("this sandbox backend does not support file reads")

    def write_file(self, session_id: str, path: str, content: str) -> str:
        """Write text (creating parent dirs). Returns the absolute path written."""
        raise NotImplementedError("this sandbox backend does not support file writes")


class LocalSubprocessSandbox(SandboxBackend):
    def __init__(self, max_output: int = 10_000, workspace: str | None = None):
        """`workspace`: a real directory every session starts in (e.g. the repo
        you launched from) so the agent can read/edit your actual project, like
        Claude Code. It is SHARED across sessions and NEVER deleted by
        `destroy()`. When None, each session gets a private throwaway tempdir
        that IS deleted on `destroy()` (the isolated default)."""
        self._dirs: dict[str, str] = {}  # session_id -> root dir
        self._cwd: dict[str, str] = {}  # session_id -> current working dir (persisted)
        self._owned: set[str] = set()  # roots we created and may safely rmtree
        self.max_output = max_output
        self.workspace = workspace

    def _root(self, session_id: str) -> str:
        if session_id not in self._dirs:
            if self.workspace:
                root = self.workspace  # shared, external: never deleted
            else:
                root = tempfile.mkdtemp(prefix=f"hsbx-{session_id[:8]}-")
                self._owned.add(root)  # private, throwaway: safe to delete
            self._dirs[session_id] = root
            self._cwd[session_id] = root
        return self._dirs[session_id]

    def _resolve(self, session_id: str, path: str) -> Path:
        """Resolve a possibly-relative path against the session's current cwd —
        the same directory `exec` runs in — so Bash/Write/Edit stay in sync."""
        self._root(session_id)
        p = Path(path).expanduser()
        return p if p.is_absolute() else Path(self._cwd[session_id]) / p

    def read_file(self, session_id: str, path: str) -> str:
        return self._resolve(session_id, path).read_text()

    def write_file(self, session_id: str, path: str, content: str) -> str:
        dest = self._resolve(session_id, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        return str(dest)

    def exec(self, session_id, command, timeout=60) -> ExecResult:
        self._root(session_id)
        cwd = self._cwd[session_id]
        # Run the command in the session's persisted cwd, then print the final cwd
        # on its own line so `cd` inside the command carries to the next call.
        # The command runs in the parent shell (not a subshell), so its cd sticks.
        wrapped = f"{command}\n__rc=$?\nprintf '\\n{_CWD_MARK}%s\\n' \"$(pwd)\"\nexit $__rc"
        try:
            p = subprocess.run(
                wrapped,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
            )
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timeout after {timeout}s", 124, cwd)
        except FileNotFoundError:  # no /bin/bash (rare) — fall back to default shell
            p = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
            )
            return ExecResult(self._cap(p.stdout), self._cap(p.stderr), p.returncode, cwd)

        stdout, new_cwd = self._extract_cwd(p.stdout, cwd)
        if Path(new_cwd).is_dir():
            self._cwd[session_id] = new_cwd
        return ExecResult(
            self._cap(stdout), self._cap(p.stderr), p.returncode, self._cwd[session_id]
        )

    @staticmethod
    def _extract_cwd(stdout: str, fallback: str) -> tuple[str, str]:
        lines = stdout.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(_CWD_MARK):
                cwd = lines[i][len(_CWD_MARK) :].strip()
                # drop the marker line (and a trailing blank we injected)
                kept = lines[:i]
                if kept and kept[-1] == "":
                    kept.pop()
                return "\n".join(kept), cwd
        return stdout, fallback

    def _cap(self, text: str) -> str:
        """Head/tail elision for very large output (mini-SWE style)."""
        if text is None:
            return ""
        if len(text) <= self.max_output:
            return text
        half = self.max_output // 2
        elided = len(text) - 2 * half
        return (
            f"{text[:half]}\n"
            f"... [{elided} characters elided; use head/tail/grep to narrow] ...\n"
            f"{text[-half:]}"
        )

    def destroy(self, session_id) -> None:
        root = self._dirs.pop(session_id, None)
        self._cwd.pop(session_id, None)
        # Only delete dirs WE created — never an external workspace (your repo).
        if root and root in self._owned and root not in self._dirs.values():
            shutil.rmtree(root, ignore_errors=True)
            self._owned.discard(root)
