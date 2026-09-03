"""Shared complete-sentence context selection for emotion conditioning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence


CONTEXT_POLICY = "nearest-three-complete-preceding-v1"
CONTEXT_SENTENCE_LIMIT = 3
CONTEXT_DELIMITER = "\n"
CONTEXT_MAX_LENGTH = 256

_SPACE = re.compile(r"\s+")


def normalize_context_sentence(value: object) -> str:
    """Trim a source sentence and collapse internal whitespace."""

    return _SPACE.sub(" ", str(value or "").strip())


@dataclass(frozen=True)
class ContextSelection:
    previous_sentences: tuple[str, ...]
    previous_text: str
    input_token_count: int
    target_token_count: int
    context_limited: bool
    target_truncated: bool
    stop_reason: str


def select_complete_context(
    previous_sentences: Sequence[object],
    count_pair_tokens: Callable[[str], int],
    *,
    max_length: int = CONTEXT_MAX_LENGTH,
    sentence_limit: int = CONTEXT_SENTENCE_LIMIT,
    delimiter: str = CONTEXT_DELIMITER,
) -> ContextSelection:
    """Select the nearest complete preceding sentences within a pair-token budget.

    ``previous_sentences`` is chronological.  The callback receives the joined
    first sequence; the target sequence is fixed by the caller.
    """

    if max_length <= 0:
        raise ValueError("max_length 必须为正数")
    if sentence_limit <= 0:
        raise ValueError("sentence_limit 必须为正数")
    normalized = tuple(
        text
        for text in (normalize_context_sentence(value) for value in previous_sentences)
        if text
    )
    candidates = normalized[-sentence_limit:]
    target_token_count = int(count_pair_tokens(""))
    if target_token_count < 0:
        raise ValueError("token 计数不能为负数")
    if target_token_count > max_length:
        return ContextSelection(
            previous_sentences=(),
            previous_text="",
            input_token_count=max_length,
            target_token_count=target_token_count,
            context_limited=bool(candidates),
            target_truncated=True,
            stop_reason="target-over-max",
        )

    selected: tuple[str, ...] = ()
    input_token_count = target_token_count
    stopped_for_budget = False
    for sentence in reversed(candidates):
        trial = (sentence, *selected)
        trial_text = delimiter.join(trial)
        trial_count = int(count_pair_tokens(trial_text))
        if trial_count < 0:
            raise ValueError("token 计数不能为负数")
        if trial_count > max_length:
            stopped_for_budget = True
            break
        selected = trial
        input_token_count = trial_count

    if stopped_for_budget:
        stop_reason = "context-over-max"
    elif len(candidates) < sentence_limit:
        stop_reason = "window-start"
    else:
        stop_reason = "context-limit"
    return ContextSelection(
        previous_sentences=selected,
        previous_text=delimiter.join(selected),
        input_token_count=input_token_count,
        target_token_count=target_token_count,
        context_limited=stopped_for_budget,
        target_truncated=False,
        stop_reason=stop_reason,
    )
