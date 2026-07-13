"""LLM model-completion contract + OpenRouter/Azure implementations + offline fake.

Not to be confused with `harness.providers` (`ToolProvider`), which composes
capabilities — MCP servers, tool bundles — into a harness. This module is the
`Provider` the harness calls to talk to a model: `complete()`/`stream()`, plus
higher-level helpers (summarize, classify_subject, induce_skills) expressed in
terms of `complete()`.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from ..settings import Config, ProviderConfig
from .registry import ProviderRegistry, register_provider


@dataclass
class ModelResult:
    message: dict  # {"role": "assistant", "content": str, "tool_calls": [...]}
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
    def complete(
        self, model: str, messages: list[dict], tools: list[dict] | None = None
    ) -> ModelResult: ...

    def available_models(self) -> list[str] | None:
        """Model ids this provider can enumerate for validation, or ``None``
        when it can't verify (callers must then accept any id rather than
        reject a real model they simply couldn't confirm)."""
        return None

    def stream(
        self, model: str, messages: list[dict], tools: list[dict] | None = None
    ) -> Generator[str, None, ModelResult]:
        """Yield assistant text deltas; return the final ModelResult.

        Consume with a next()-loop and read the return value off StopIteration:

            gen = provider.stream(...)
            while True:
                try: delta = next(gen)
                except StopIteration as stop:
                    res = stop.value
                    break

        Default is a non-streaming fallback (one delta = the whole text), so any
        provider works without implementing real streaming. complete() is left
        untouched and continues to back summarize/classify/induce.
        """
        res = self.complete(model, messages, tools)
        if res.text:
            yield res.text
        return res

    # ---- helpers built on top of complete() ----
    def summarize(self, model: str, prev_summary: str | None, messages: list[dict]) -> str:
        head = (
            "You compress conversation history. Produce a faithful, compact "
            "summary that preserves decisions, facts, open tasks and user "
            "preferences. Fold the PREVIOUS SUMMARY into the new one."
        )
        prev = f"PREVIOUS SUMMARY:\n{prev_summary}\n\n" if prev_summary else ""
        body = "\n".join(f"[{m.get('role')}] {_text(m.get('content'))}" for m in messages)
        res = self.complete(
            model,
            [
                {"role": "system", "content": head},
                {
                    "role": "user",
                    "content": f"{prev}MESSAGES TO FOLD:\n{body}\n\nReturn only the summary.",
                },
            ],
        )
        return res.text.strip()

    def classify_subject(self, model: str, messages: list[dict]) -> str:
        body = "\n".join(_text(m.get("content")) for m in messages)
        res = self.complete(
            model,
            [
                {
                    "role": "system",
                    "content": "Classify the subject of these user "
                    "messages in at most 5 words. Return only the label.",
                },
                {"role": "user", "content": body},
            ],
        )
        return res.text.strip()[:120]

    def induce_skills(self, model: str, signals: str) -> list[dict]:
        res = self.complete(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "You detect recurring request patterns that could become reusable "
                        "skills. Return a JSON array (possibly empty) of objects with keys "
                        "name, summary (one line), body (the procedure). Return ONLY JSON."
                    ),
                },
                {"role": "user", "content": signals},
            ],
        )
        return _parse_json_array(res.text)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _chunk(text: str, n: int) -> list[str]:
    """Split `text` into at most `n` roughly equal, order-preserving pieces."""
    if not text:
        return []
    size = max(1, -(-len(text) // n))  # ceil
    return [text[i : i + size] for i in range(0, len(text), size)]


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
        return [d for d in data if isinstance(d, dict) and "name" in d]
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(Provider):
    """Shared base for any OpenAI-wire-compatible chat-completions API.

    Subclasses vary only in transport/auth (`_client`), optional query params
    (`_params`), and how they report a model's context window
    (`model_context_window`). The request/response and SSE-streaming logic —
    identical across OpenRouter, Azure AI Foundry, and vanilla OpenAI — lives
    here once.
    """

    def _client(self):
        """Return a configured httpx.Client (base_url + auth headers)."""
        raise NotImplementedError

    def _params(self) -> dict:
        """Extra query params sent with each request (e.g. api-version)."""
        return {}

    @staticmethod
    def _check(r: Any) -> None:
        """Raise a helpful error on a non-2xx response, surfacing the API's own
        message instead of a bare status line. Reads the body first so it also
        works for a streamed response. Names Azure content-filter blocks (a 400
        that otherwise looks like a mysterious bug) and which category tripped."""
        if r.is_success:
            return
        try:
            r.read()  # streamed responses must be read before .json()
            detail = r.json()
        except Exception:
            detail = None
        err = detail.get("error") if isinstance(detail, dict) else None
        if isinstance(err, dict):
            if err.get("code") == "content_filter":
                inner = err.get("innererror") or {}
                results = inner.get("content_filter_result") or {}
                cats = ", ".join(
                    k for k, v in results.items() if isinstance(v, dict) and v.get("filtered")
                )
                where = f" ({cats})" if cats else ""
                raise RuntimeError(
                    f"Request blocked by the provider's content filter{where}. "
                    "This is a safety filter on the prompt text, not a code error — "
                    "rephrase the wording and retry."
                )
            if err.get("message"):
                raise RuntimeError(f"{r.status_code} from {r.request.url}: {err['message']}")
        raise RuntimeError(f"{r.status_code} from {r.request.url}")

    def complete(self, model, messages, tools=None) -> ModelResult:
        payload: dict = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        with self._client() as c:
            r = c.post("/chat/completions", json=payload, params=self._params())
            self._check(r)
            d = r.json()
        msg = d["choices"][0]["message"]
        usage = d.get("usage", {})
        return ModelResult(
            message=msg,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    def stream(self, model, messages, tools=None) -> Generator[str, None, ModelResult]:
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}  # index -> assembled tool_call
        tokens_in = tokens_out = 0
        with self._client() as c:
            with c.stream("POST", "/chat/completions", json=payload, params=self._params()) as r:
                self._check(r)
                for raw in r.iter_lines():
                    line = raw.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage")
                    if usage:
                        tokens_in = usage.get("prompt_tokens", tokens_in)
                        tokens_out = usage.get("completion_tokens", tokens_out)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        yield piece
                    for tc in delta.get("tool_calls") or []:
                        self._merge_tool_call(tool_calls, tc)
        msg: dict = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return ModelResult(message=msg, tokens_in=tokens_in, tokens_out=tokens_out)

    @staticmethod
    def _merge_tool_call(acc: dict[int, dict], frag: dict) -> None:
        """Fold a streamed tool_call fragment into the accumulator by index.

        OpenAI/OpenRouter stream tool calls across chunks: the first carries id +
        function.name, later chunks append function.arguments string fragments.
        """
        idx = frag.get("index", 0)
        cur = acc.setdefault(
            idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if frag.get("id"):
            cur["id"] = frag["id"]
        if frag.get("type"):
            cur["type"] = frag["type"]
        fn = frag.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            cur["function"]["arguments"] += fn["arguments"]


# ---------------------------------------------------------------------------
@register_provider("openrouter")
class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._models: dict[str, dict] = {}

    def _client(self):
        import httpx  # optional dependency

        return httpx.Client(
            base_url=self.cfg.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
                "HTTP-Referer": "https://localhost",
                "X-Title": "Harness",
            },
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

    def available_models(self) -> list[str] | None:
        try:
            self._load_models()
            return list(self._models)
        except Exception:
            return None  # network/registry failure: can't verify, don't reject


# ---------------------------------------------------------------------------
@register_provider("azure")
class AzureFoundryProvider(OpenAICompatibleProvider):
    """Azure AI Foundry (OpenAI-compatible v1 endpoint).

    Auth precedence: a static API key (`AZURE_AI_API_KEY`) if set, otherwise a
    Microsoft Entra ID bearer token via `azure-identity` (managed identity in
    Azure, `DefaultAzureCredential` for local dev). Tokens are fetched at
    runtime and cached in memory by the SDK — nothing is written to disk, so
    this is safe in an ephemeral container.
    """

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        # lazily built azure-identity callable that returns a bearer token
        self._token_provider: Callable[[], str] | None = None

    def _bearer(self) -> str:
        provider = self._token_provider
        if provider is None:
            try:
                from azure.identity import (
                    DefaultAzureCredential,
                    ManagedIdentityCredential,
                    get_bearer_token_provider,
                )
            except ImportError as e:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "Azure AI Foundry without AZURE_AI_API_KEY needs managed "
                    "identity: install the 'azure' extra (pip install "
                    "'harness[azure]' / azure-identity)."
                ) from e
            import os

            # In Azure (Container Apps sets IDENTITY_ENDPOINT) prefer the
            # managed identity directly; locally fall back to the dev chain.
            if os.environ.get("IDENTITY_ENDPOINT") or os.environ.get("MSI_ENDPOINT"):
                cred = ManagedIdentityCredential(client_id=self.cfg.azure_client_id or None)
            else:
                cred = DefaultAzureCredential()
            provider = get_bearer_token_provider(
                cred, "https://cognitiveservices.azure.com/.default"
            )
            self._token_provider = provider
        return provider()

    def _auth_headers(self) -> dict:
        if self.cfg.azure_api_key:
            return {"api-key": self.cfg.azure_api_key}
        return {"Authorization": f"Bearer {self._bearer()}"}

    def _client(self):
        import httpx  # optional dependency

        base = self.cfg.azure_endpoint.rstrip("/") + "/openai/v1"
        return httpx.Client(base_url=base, headers=self._auth_headers(), timeout=120)

    def _params(self) -> dict:
        return {"api-version": self.cfg.azure_api_version} if self.cfg.azure_api_version else {}

    def model_context_window(self, model: str) -> int:
        # Azure AI Foundry has no OpenRouter-style /models context_length
        # endpoint; report the configured fallback.
        return self.cfg.default_context_window


# ---------------------------------------------------------------------------
@register_provider("vertex")
class VertexProvider(OpenAICompatibleProvider):
    """Google Vertex AI via its OpenAI-compatible chat-completions endpoint.

    Because the wire format is OpenAI-compatible, all request/response and
    SSE-streaming logic is inherited from `OpenAICompatibleProvider`; only the
    transport (base URL + a short-lived OAuth bearer) differs.

    Auth uses Application Default Credentials (`google-auth`): a service-account
    key file (`GOOGLE_APPLICATION_CREDENTIALS`), `gcloud auth application-default
    login`, or workload identity. Tokens are fetched at runtime and refreshed in
    memory — nothing is written to disk, so this is safe in an ephemeral
    container. Model ids look like `google/gemini-2.0-flash`.
    """

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._credentials = None  # lazily built google.auth credentials

    def _bearer(self) -> str:
        creds = self._credentials
        if creds is None:
            try:
                import google.auth
                from google.auth.transport.requests import Request  # noqa: F401
            except ImportError as e:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "Vertex AI needs Application Default Credentials: install the "
                    "'vertex' extra (pip install 'harness[vertex]' / google-auth)."
                ) from e
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._credentials = creds
        if not creds.valid:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        return creds.token

    def _client(self):
        import httpx  # optional dependency

        loc, proj = self.cfg.vertex_location, self.cfg.vertex_project
        base = (
            f"https://{loc}-aiplatform.googleapis.com/v1/projects/{proj}"
            f"/locations/{loc}/endpoints/openapi"
        )
        return httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {self._bearer()}"},
            timeout=120,
        )

    def model_context_window(self, model: str) -> int:
        # The Vertex OpenAI-compat endpoint exposes no models listing; report
        # the configured fallback.
        return self.cfg.default_context_window


# ---------------------------------------------------------------------------
@register_provider("bedrock")
class BedrockProvider(Provider):
    """AWS Bedrock via the unified **Converse** API (`boto3`).

    Bedrock is NOT OpenAI-wire-compatible, so this subclasses `Provider`
    directly and translates between the harness's OpenAI-shaped messages/tools
    and Bedrock's Converse schema (both directions, streaming and not).

    Auth is the standard boto3 credential chain (env vars, shared config/profile,
    or an IAM role) — nothing is stored in config. `model` is a Bedrock model id
    or inference-profile id, e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0` or
    `us.anthropic.claude-3-5-sonnet-20241022-v2:0`.
    """

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._runtime = None  # lazily built bedrock-runtime client

    def _client(self):
        rt = self._runtime
        if rt is None:
            try:
                import boto3  # optional dependency
            except ImportError as e:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "AWS Bedrock needs boto3: install the 'bedrock' extra "
                    "(pip install 'harness[bedrock]' / boto3)."
                ) from e
            rt = boto3.client("bedrock-runtime", region_name=self.cfg.bedrock_region or None)
            self._runtime = rt
        return rt

    # --- OpenAI <-> Converse translation ---------------------------------
    @staticmethod
    def _to_converse(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split OpenAI-shaped messages into Converse (system_blocks, messages).

        Consecutive same-role turns are merged, because Converse requires strict
        user/assistant alternation and folds tool results into user turns.
        """
        system: list[dict] = []
        conv: list[dict] = []

        def emit(role: str, blocks: list[dict]) -> None:
            if not blocks:
                return
            if conv and conv[-1]["role"] == role:
                conv[-1]["content"].extend(blocks)
            else:
                conv.append({"role": role, "content": list(blocks)})

        for m in messages:
            role = m.get("role")
            if role == "system":
                text = _text(m.get("content"))
                if text:
                    system.append({"text": text})
            elif role == "user":
                emit("user", [{"text": _text(m.get("content"))}])
            elif role == "assistant":
                blocks: list[dict] = []
                text = _text(m.get("content")) if m.get("content") else ""
                if text:
                    blocks.append({"text": text})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": _loads(fn.get("arguments")),
                            }
                        }
                    )
                emit("assistant", blocks)
            elif role == "tool":
                emit(
                    "user",
                    [
                        {
                            "toolResult": {
                                "toolUseId": m.get("tool_call_id", ""),
                                "content": [{"text": _text(m.get("content"))}],
                            }
                        }
                    ],
                )
        return system, conv

    @staticmethod
    def _tool_config(tools: list[dict] | None) -> dict | None:
        if not tools:
            return None
        specs = []
        for t in tools:
            fn = t.get("function", t)
            specs.append(
                {
                    "toolSpec": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "inputSchema": {
                            "json": fn.get("parameters") or {"type": "object", "properties": {}}
                        },
                    }
                }
            )
        return {"tools": specs}

    def _request(self, model: str, messages: list[dict], tools: list[dict] | None) -> dict:
        system, conv = self._to_converse(messages)
        kwargs: dict = {"modelId": model, "messages": conv}
        if system:
            kwargs["system"] = system
        tc = self._tool_config(tools)
        if tc:
            kwargs["toolConfig"] = tc
        return kwargs

    @staticmethod
    def _assistant_message(content_parts: list[str], tool_calls: list[dict]) -> dict:
        msg: dict = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def complete(self, model, messages, tools=None) -> ModelResult:
        resp = self._client().converse(**self._request(model, messages, tools))
        out = resp.get("output", {}).get("message", {})
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in out.get("content", []):
            if "text" in block:
                content_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tu.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                )
        usage = resp.get("usage", {})
        return ModelResult(
            message=self._assistant_message(content_parts, tool_calls),
            tokens_in=usage.get("inputTokens", 0),
            tokens_out=usage.get("outputTokens", 0),
        )

    def stream(self, model, messages, tools=None) -> Generator[str, None, ModelResult]:
        resp = self._client().converse_stream(**self._request(model, messages, tools))
        content_parts: list[str] = []
        tools_by_index: dict[int, dict] = {}  # contentBlockIndex -> {id,name,input}
        tokens_in = tokens_out = 0
        for event in resp["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                tu = (start.get("start") or {}).get("toolUse")
                if tu:
                    tools_by_index[start["contentBlockIndex"]] = {
                        "id": tu.get("toolUseId", ""),
                        "name": tu.get("name", ""),
                        "input": "",
                    }
            elif "contentBlockDelta" in event:
                block = event["contentBlockDelta"]
                delta = block.get("delta", {})
                if "text" in delta:
                    content_parts.append(delta["text"])
                    yield delta["text"]
                elif "toolUse" in delta:
                    acc = tools_by_index.setdefault(
                        block["contentBlockIndex"], {"id": "", "name": "", "input": ""}
                    )
                    acc["input"] += delta["toolUse"].get("input", "")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                tokens_in = usage.get("inputTokens", tokens_in)
                tokens_out = usage.get("outputTokens", tokens_out)
        tool_calls = [
            {
                "id": tools_by_index[i]["id"],
                "type": "function",
                "function": {
                    "name": tools_by_index[i]["name"],
                    "arguments": tools_by_index[i]["input"] or "{}",
                },
            }
            for i in sorted(tools_by_index)
        ]
        return ModelResult(
            message=self._assistant_message(content_parts, tool_calls),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def model_context_window(self, model: str) -> int:
        # Bedrock's runtime API doesn't report context windows; use the fallback.
        return self.cfg.default_context_window

    def available_models(self) -> list[str] | None:
        # Enumerate base foundation models AND cross-region inference profiles,
        # since Converse accepts either id. Any failure (permissions, region) ->
        # None so the caller accepts an unverifiable-but-valid id rather than
        # falsely rejecting it.
        import contextlib

        try:
            import boto3

            ctl = boto3.client("bedrock", region_name=self.cfg.bedrock_region or None)
            ids = [m["modelId"] for m in ctl.list_foundation_models().get("modelSummaries", [])]
            with contextlib.suppress(Exception):  # inference profiles optional / may be denied
                ids += [
                    p["inferenceProfileId"]
                    for p in ctl.list_inference_profiles().get("inferenceProfileSummaries", [])
                ]
            return ids or None
        except Exception:
            return None


def _loads(raw: Any) -> dict:
    """Parse a tool-call arguments JSON string into a dict (Converse wants an
    object, not a string). Tolerates empty/malformed input."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
class FakeProvider(Provider):
    """Deterministic, network-free provider for the offline demo and tests.

    Drive its behaviour with a queued script of responses; otherwise it echoes.
    """

    def __init__(self, context_window: int = 2000):
        self._ctx = context_window
        self.script: list[dict] = []  # queued assistant messages
        self.calls: list[list[dict]] = []

    def model_context_window(self, model: str) -> int:
        return self._ctx

    def queue(self, content: str = "", tool_calls: list[dict] | None = None) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
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

    def stream(self, model, messages, tools=None) -> Generator[str, None, ModelResult]:
        """Same script/echo behaviour as complete(), but the content is emitted
        in a few chunks so the offline demo and tests visibly stream. Token
        accounting is identical to complete(), so run_turn's TurnResult is
        unchanged under streaming."""
        self.calls.append(messages)
        if self.script:
            msg = self.script.pop(0)
        else:
            last = _text(messages[-1].get("content")) if messages else ""
            msg = {"role": "assistant", "content": f"OK: {last[:80]}"}
        ti = sum(len(_text(m.get("content"))) for m in messages) // 4
        to = len(_text(msg.get("content"))) // 4
        text = msg.get("content") or ""
        yield from _chunk(text, 3)
        return ModelResult(message=msg, tokens_in=max(1, ti), tokens_out=max(1, to))

    # cheap deterministic helpers (avoid consuming the scripted queue)
    def summarize(self, model, prev_summary, messages) -> str:
        prefix = (prev_summary + " | ") if prev_summary else ""
        return f"{prefix}summary of {len(messages)} msgs"

    def classify_subject(self, model, messages) -> str:
        return "topic-" + str(len(messages))

    def induce_skills(self, model, signals) -> list[dict]:
        return []


def _build_fake(cfg: ProviderConfig) -> FakeProvider:
    return FakeProvider(context_window=cfg.default_context_window)


ProviderRegistry.register("fake", _build_fake)


def build_provider(cfg: Config) -> Provider:
    """Explicit-name lookup only — performs NO detection itself.

    The caller supplies `cfg.provider.name` (from a config file, a CLI flag,
    their own env-reading code, or a per-tenant setting); this just resolves
    that name through `ProviderRegistry`. If you don't want to manage a name,
    construct a provider directly and pass `provider=...` to `Harness`, or use
    `detect_provider()` for the same env-sniffing convenience the old default
    used to apply silently.
    """
    if not cfg.provider.name:
        raise ValueError(
            "cfg.provider.name must be set to build a provider by name; "
            "otherwise construct one directly and pass provider=... to Harness."
        )
    return ProviderRegistry.build(cfg.provider.name, cfg.provider)


def detect_provider(cfg: Config) -> Provider:
    """Convenience, opt-in heuristic for demos/CLIs that don't want to manage
    `cfg.provider.name`. Precedence: Azure (`azure_endpoint`) → Vertex
    (`vertex_project`) → Bedrock (`bedrock_region`) → OpenRouter
    (`openrouter_api_key`) → offline `FakeProvider`.

    This is NOT called by `Harness` or by `build_provider` — it exists purely
    for application code (see `harness/cli.py`) that wants the old "just pick
    something reasonable from what's configured" behavior explicitly.
    """
    p = cfg.provider
    if p.azure_endpoint:
        return AzureFoundryProvider(p)
    if p.vertex_project:
        return VertexProvider(p)
    if p.bedrock_region:
        return BedrockProvider(p)
    if p.openrouter_api_key:
        return OpenRouterProvider(p)
    return FakeProvider(context_window=p.default_context_window)


def provider_label(cfg: Config) -> str:
    """Human-readable name of the provider `detect_provider` would select."""
    p = cfg.provider
    if p.azure_endpoint:
        return "Azure AI Foundry"
    if p.vertex_project:
        return "Google Vertex AI"
    if p.bedrock_region:
        return "AWS Bedrock"
    if p.openrouter_api_key:
        return "OpenRouter"
    return "FakeProvider (offline)"
