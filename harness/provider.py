"""LLM provider contract + OpenRouter implementation + offline fake.

The harness depends only on `Provider`. Higher-level helpers (summarize,
classify_subject, induce_skills) are expressed in terms of `complete()`.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Config


@dataclass
class ModelResult:
    message: dict           # {"role": "assistant", "content": str, "tool_calls": [...]}
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []

    @property
    def text(self) -> str:
        return self.message.get("content") or ""


class Provider(abc.ABC):
    @abc.abstractmethod
    def model_context_window(self, model: str) -> int: ...

    @abc.abstractmethod
    def complete(self, model: str, messages: list[dict],
                 tools: Optional[list[dict]] = None) -> ModelResult: ...

    # ---- helpers built on top of complete() ----
    def summarize(self, model: str, prev_summary: Optional[str],
                  messages: list[dict]) -> str:
        head = ("You compress conversation history. Produce a faithful, compact "
                "summary that preserves decisions, facts, open tasks and user "
                "preferences. Fold the PREVIOUS SUMMARY into the new one.")
        prev = f"PREVIOUS SUMMARY:\n{prev_summary}\n\n" if prev_summary else ""
        body = "\n".join(f"[{m.get('role')}] {_text(m.get('content'))}" for m in messages)
        res = self.complete(model, [
            {"role": "system", "content": head},
            {"role": "user", "content": f"{prev}MESSAGES TO FOLD:\n{body}\n\nReturn only the summary."},
        ])
        return res.text.strip()

    def classify_subject(self, model: str, messages: list[dict]) -> str:
        body = "\n".join(_text(m.get("content")) for m in messages)
        res = self.complete(model, [
            {"role": "system", "content": "Classify the subject of these user "
             "messages in at most 5 words. Return only the label."},
            {"role": "user", "content": body},
        ])
        return res.text.strip()[:120]

    def induce_skills(self, model: str, signals: str) -> list[dict]:
        res = self.complete(model, [
            {"role": "system", "content": (
                "You detect recurring request patterns that could become reusable "
                "skills. Return a JSON array (possibly empty) of objects with keys "
                "name, summary (one line), body (the procedure). Return ONLY JSON.")},
            {"role": "user", "content": signals},
        ])
        return _parse_json_array(res.text)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
        return [d for d in data if isinstance(d, dict) and "name" in d]
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
class OpenRouterProvider(Provider):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._models: dict[str, dict] = {}

    def _client(self):
        import httpx  # optional dependency
        return httpx.Client(
            base_url=self.cfg.openrouter_base_url,
            headers={"Authorization": f"Bearer {self.cfg.openrouter_api_key}",
                     "HTTP-Referer": "https://localhost", "X-Title": "Harness"},
            timeout=120,
        )

    def _load_models(self) -> None:
        if self._models:
            return
        with self._client() as c:
            data = c.get("/models").json()["data"]
        self._models = {m["id"]: m for m in data}

    def model_context_window(self, model: str) -> int:
        try:
            self._load_models()
            m = self._models.get(model, {})
            return int(m.get("context_length") or self.cfg.default_context_window)
        except Exception:
            return self.cfg.default_context_window

    def complete(self, model, messages, tools=None) -> ModelResult:
        payload: dict = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        with self._client() as c:
            r = c.post("/chat/completions", json=payload)
            r.raise_for_status()
            d = r.json()
        msg = d["choices"][0]["message"]
        usage = d.get("usage", {})
        return ModelResult(message=msg,
                           tokens_in=usage.get("prompt_tokens", 0),
                           tokens_out=usage.get("completion_tokens", 0))


# ---------------------------------------------------------------------------
class FakeProvider(Provider):
    """Deterministic, network-free provider for the offline demo and tests.

    Drive its behaviour with a queued script of responses; otherwise it echoes.
    """

    def __init__(self, context_window: int = 2000):
        self._ctx = context_window
        self.script: list[dict] = []     # queued assistant messages
        self.calls: list[list[dict]] = []

    def model_context_window(self, model: str) -> int:
        return self._ctx

    def queue(self, content: str = "", tool_calls: Optional[list[dict]] = None) -> None:
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.script.append(msg)

    def complete(self, model, messages, tools=None) -> ModelResult:
        self.calls.append(messages)
        if self.script:
            msg = self.script.pop(0)
        else:
            last = _text(messages[-1].get("content")) if messages else ""
            msg = {"role": "assistant", "content": f"OK: {last[:80]}"}
        ti = sum(len(_text(m.get("content"))) for m in messages) // 4
        to = len(_text(msg.get("content"))) // 4
        return ModelResult(message=msg, tokens_in=max(1, ti), tokens_out=max(1, to))

    # cheap deterministic helpers (avoid consuming the scripted queue)
    def summarize(self, model, prev_summary, messages) -> str:
        prefix = (prev_summary + " | ") if prev_summary else ""
        return f"{prefix}summary of {len(messages)} msgs"

    def classify_subject(self, model, messages) -> str:
        return "topic-" + str(len(messages))

    def induce_skills(self, model, signals) -> list[dict]:
        return []


def build_provider(cfg: Config) -> Provider:
    if cfg.openrouter_api_key:
        return OpenRouterProvider(cfg)
    return FakeProvider(context_window=cfg.default_context_window)
