"""Observability: every loop step is logged; token spend is first-class."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from .repository import Repository


class Observer:
    def __init__(self, repo: Repository, echo: bool = False):
        self.repo = repo
        self.echo = echo

    def log(self, session_id: Optional[str], turn_id: Optional[str], step_type: str,
            detail: dict, tokens_in: Optional[int] = None,
            tokens_out: Optional[int] = None, latency_ms: Optional[int] = None) -> None:
        self.repo.add_step_log(session_id, turn_id, step_type, detail,
                               tokens_in, tokens_out, latency_ms)
        if self.echo:
            cost = ""
            if tokens_in is not None or tokens_out is not None:
                cost = f" [in={tokens_in or 0} out={tokens_out or 0}]"
            print(f"  · {step_type}{cost} {detail}")

    @contextmanager
    def timed(self, session_id, turn_id, step_type, detail):
        """Context manager that records latency for a step. Yields a dict that
        the caller can fill with tokens_in/tokens_out before exit."""
        slot = {"tokens_in": None, "tokens_out": None}
        t0 = time.perf_counter()
        try:
            yield slot
        finally:
            dt = int((time.perf_counter() - t0) * 1000)
            self.log(session_id, turn_id, step_type, detail,
                     slot["tokens_in"], slot["tokens_out"], dt)
