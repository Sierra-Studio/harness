"""Execution backend for the Bash tool.

`SandboxBackend` is the pluggable contract. `LocalSubprocessSandbox` is the
trivial Phase-2 implementation: one working directory per session, command
timeout, output capped. It is NOT isolated — swap it for a kernel-isolated
backend (gVisor / Firecracker / K8s Pod) for real multi-tenant safety. See the
companion architecture guide.
"""
from __future__ import annotations

import abc
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxBackend(abc.ABC):
    @abc.abstractmethod
    def exec(self, session_id: str, command: str, timeout: int = 60) -> ExecResult: ...

    @abc.abstractmethod
    def destroy(self, session_id: str) -> None: ...


class LocalSubprocessSandbox(SandboxBackend):
    def __init__(self, max_output: int = 8000):
        self._dirs: dict[str, str] = {}
        self.max_output = max_output

    def _workdir(self, session_id: str) -> str:
        if session_id not in self._dirs:
            self._dirs[session_id] = tempfile.mkdtemp(prefix=f"hsbx-{session_id[:8]}-")
        return self._dirs[session_id]

    def exec(self, session_id, command, timeout=60) -> ExecResult:
        wd = self._workdir(session_id)
        try:
            p = subprocess.run(
                command, shell=True, cwd=wd, capture_output=True, text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=p.stdout[: self.max_output],
                stderr=p.stderr[: self.max_output],
                exit_code=p.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr=f"timeout after {timeout}s", exit_code=124)

    def destroy(self, session_id) -> None:
        wd = self._dirs.pop(session_id, None)
        if wd:
            shutil.rmtree(wd, ignore_errors=True)
