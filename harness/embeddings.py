"""Embedding provider for semantic search over tool_index and skills.

Uses an OpenAI-compatible endpoint when EMBEDDING_API_KEY is set; otherwise a
deterministic LOCAL fallback (hashing token features into a fixed-dim vector).
The fallback is NOT semantic — it only keeps the pipeline runnable offline.
"""
from __future__ import annotations

import hashlib
import math
import re

from .config import Config

_WORD = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = cfg.embedding_dim
        self._remote = bool(cfg.embedding_api_key)

    def embed(self, text: str) -> list[float]:
        if self._remote:
            try:
                return self._embed_remote(text)
            except Exception:
                # fall back rather than crash the harness
                pass
        return self._embed_local(text)

    # --- remote (OpenAI-compatible) ---
    def _embed_remote(self, text: str) -> list[float]:
        import httpx  # local import; optional dependency

        r = httpx.post(
            f"{self.cfg.embedding_base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.cfg.embedding_api_key}"},
            json={"model": self.cfg.embedding_model, "input": text},
            timeout=30,
        )
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        return _normalize(vec)

    # --- local deterministic fallback ---
    def _embed_local(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return _normalize(vec)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are normalized
