"""Token counting. Uses tiktoken when available, else a heuristic fallback.

The fallback is an APPROXIMATION used only for fitting content into the window.
The authoritative token spend always comes from the provider's `usage` field.
"""
from __future__ import annotations

import json
from typing import Any

try:  # pragma: no cover - depends on environment
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def count_tokens(content: Any) -> int:
    """Count tokens for a string or any JSON-serializable message content."""
    text = _to_text(content)
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Heuristic: ~4 chars/token, with a small floor so empty content isn't zero.
    return max(1, (len(text) + 3) // 4)


def count_messages(messages: list[dict]) -> int:
    """Approximate tokens for a list of chat messages (content + small overhead)."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", "")) + 4  # per-message overhead
        for call in m.get("tool_calls", []) or []:
            total += count_tokens(call)
    return total
