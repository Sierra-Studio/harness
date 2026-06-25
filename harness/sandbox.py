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


class LocalSubprocessSandbox(SandboxBackend):
    def __init__(self, max_output: int = 10_000):
        self._dirs: dict[str, str] = {}   # session_id -> root tempdir
        self._cwd: dict[str, str] = {}    # session_id -> current working dir (persisted)
        self.max_output = max_output

    def _root(self, session_id: str) -> str:
        if session_id not in self._dirs:
            root = tempfile.mkdtemp(prefix=f"hsbx-{session_id[:8]}-")
            self._dirs[session_id] = root
            self._cwd[session_id] = root
        return self._dirs[session_id]

    def exec(self, session_id, command, timeout=60) -> ExecResult:
        self._root(session_id)
        cwd = self._cwd[session_id]
        # Run the command in the session's persisted cwd, then print the final cwd
        # on its own line so `cd` inside the command carries to the next call.
        # The command runs in the parent shell (not a subshell), so its cd sticks.
        wrapped = f"{command}\n__rc=$?\nprintf '\\n{_CWD_MARK}%s\\n' \"$(pwd)\"\nexit $__rc"
        try:
            p = subprocess.run(
                wrapped, shell=True, cwd=cwd, capture_output=True, text=True,
                timeout=timeout, executable="/bin/bash",
            )
        except subprocess.TimeoutExpired:
            return ExecResult("", f"timeout after {timeout}s", 124, cwd)
        except FileNotFoundError:  # no /bin/bash (rare) — fall back to default shell
            p = subprocess.run(command, shell=True, cwd=cwd, capture_output=True,
                               text=True, timeout=timeout)
            return ExecResult(self._cap(p.stdout), self._cap(p.stderr),
                              p.returncode, cwd)

        stdout, new_cwd = self._extract_cwd(p.stdout, cwd)
        if Path(new_cwd).is_dir():
            self._cwd[session_id] = new_cwd
        return ExecResult(self._cap(stdout), self._cap(p.stderr), p.returncode,
                          self._cwd[session_id])

    @staticmethod
    def _extract_cwd(stdout: str, fallback: str) -> tuple[str, str]:
        lines = stdout.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(_CWD_MARK):
                cwd = lines[i][len(_CWD_MARK):].strip()
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
        return (f"{text[:half]}\n"
                f"... [{elided} characters elided; use head/tail/grep to narrow] ...\n"
                f"{text[-half:]}")

    def destroy(self, session_id) -> None:
        root = self._dirs.pop(session_id, None)
        self._cwd.pop(session_id, None)
        if root:
            shutil.rmtree(root, ignore_errors=True)
