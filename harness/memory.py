"""Memory: token-budgeted context window, chained summarization, checkpoints.

Window = system prompt + accumulated (chained) summary + active turns.
Budget = context_window - system_prompt_tokens - response_reserve.
On overflow: keep the last SUMMARY_KEEP_RATIO turns verbatim, fold the rest
(plus the previous summary) into a new chained summary. Folded turns stay in
the DB with in_window=false — nothing is lost.
"""
from __future__ import annotations

import json
from math import ceil
from typing import Any, Optional

from .config import Config
from .models import Session, Turn
from .observer import Observer
from .provider import Provider
from .repository import Repository
from .tokenizer import count_tokens


class Memory:
    def __init__(self, repo: Repository, provider: Provider, cfg: Config,
                 observer: Observer):
        self.repo = repo
        self.provider = provider
        self.cfg = cfg
        self.observer = observer

    # ---- writing ----
    def append(self, session: Session, role: str, content: Any,
               tokens_in: Optional[int] = None,
               tokens_out: Optional[int] = None) -> Turn:
        return self.repo.add_turn(
            session.id, session.user_id, role, content,
            token_count=count_tokens(content),
            tokens_in=tokens_in, tokens_out=tokens_out)

    # ---- budget ----
    def budget(self, session: Session, system_prompt: str) -> int:
        sys_tokens = count_tokens(system_prompt)
        return max(0, session.context_window - sys_tokens
                   - self.cfg.response_reserve_tokens)

    # ---- reading: assemble the active window as chat messages ----
    def build_window(self, session: Session, system_prompt: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": system_prompt}]
        summ = self.repo.current_summary(session.id)
        if summ:
            msgs.append({"role": "system",
                         "content": "Conversation summary so far:\n" + summ.content})
        for t in self.repo.active_turns(session.id):
            msgs.append(self._as_message(t))
        return msgs

    @staticmethod
    def _as_message(turn: Turn) -> dict:
        if turn.role == "assistant":
            return turn.content if isinstance(turn.content, dict) else \
                {"role": "assistant", "content": str(turn.content)}
        if turn.role == "tool":
            c = turn.content
            if isinstance(c, dict):
                return {"role": "tool", "tool_call_id": c.get("tool_call_id"),
                        "content": c.get("content", "")}
            return {"role": "tool", "content": str(c)}
        if turn.role == "user" and isinstance(turn.content, dict) \
                and turn.content.get("kind") == "user_message":
            # Attachment-bearing user turn (app envelope): the model sees the
            # `content` field verbatim — a plain string, or a multimodal block
            # list (text + image_url) for vision. The sibling display metadata
            # (text/attachments) is for the UI only and is dropped here.
            return {"role": "user", "content": turn.content["content"]}
        content = turn.content if isinstance(turn.content, str) \
            else json.dumps(turn.content, ensure_ascii=False)
        return {"role": turn.role, "content": content}

    # ---- summarization ----
    def maybe_summarize(self, session: Session, system_prompt: str) -> bool:
        budget = self.budget(session, system_prompt)
        active = self.repo.active_turns(session.id)
        summ = self.repo.current_summary(session.id)
        summ_tokens = summ.token_count if summ else 0
        window = summ_tokens + sum(t.token_count for t in active)
        if window < budget or not active:
            return False

        keep_n = max(1, ceil(self.cfg.summary_keep_ratio * len(active)))
        to_fold = active[:-keep_n]
        if not to_fold:               # everything is within the kept slice
            return False

        fold_msgs = [self._as_message(t) for t in to_fold]
        prev_text = summ.content if summ else None
        new_text = self.provider.summarize(session.model, prev_text, fold_msgs)

        self.repo.add_summary(
            session.id, parent_id=summ.id if summ else None,
            content=new_text, token_count=count_tokens(new_text),
            covers_until=to_fold[-1].idx)
        self.repo.mark_out_of_window([t.id for t in to_fold])
        self.observer.log(session.id, None, "summarize",
                          {"folded": len(to_fold), "kept": keep_n})
        return True

    # ---- checkpoints ----
    def maybe_checkpoint(self, session: Session) -> bool:
        n = self.repo.count_user_turns(session.id)
        every = self.cfg.checkpoint_every_user_turns
        if n == 0 or n % every != 0:
            return False
        last = self.repo.last_checkpoint_turn(session.id)
        if last >= n:
            return False
        recent = self.repo.user_turns_since(session.id, last)
        msgs = [self._as_message(t) for t in recent]
        label = self.provider.classify_subject(session.model, msgs)
        self.repo.add_checkpoint(session.id, session.user_id, n, label)
        self.observer.log(session.id, None, "checkpoint", {"label": label, "at": n})
        return True
