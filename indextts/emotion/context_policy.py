"""Shared context policies for Readest emotion conditioning.

The production ABI uses a single target-marked sequence. The legacy
``select_complete_context`` helper remains available only so historical
three-preceding-sentence snapshots can still be audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


EMOTION_ABI = "readest-emotion-v3"
CONTEXT_POLICY = "target-prev-next-same-line-backfill-prev-section-v1"
CONTEXT_ENCODING = "tgt-type-marked-single-sequence-v1"
CONTEXT_MAX_LENGTH = 512
TGT_OPEN = "[TGT]"
TGT_CLOSE = "[/TGT]"
NARRATION_TOKEN = "[旁白]"
DIALOGUE_TOKEN = "[对白]"
NL_TOKEN = "[NL]"
EMOTION_SPECIAL_TOKENS = (
    TGT_OPEN,
    TGT_CLOSE,
    NARRATION_TOKEN,
    DIALOGUE_TOKEN,
    NL_TOKEN,
)

# Historical context3 snapshot constants. New training/runtime code must not
# use these as the production conditioning contract.
LEGACY_CONTEXT_POLICY = "nearest-three-complete-preceding-v1"
LEGACY_CONTEXT_SENTENCE_LIMIT = 3
LEGACY_CONTEXT_DELIMITER = "\n"

_SPACE = re.compile(r"\s+")


def normalize_context_sentence(value: object) -> str:
    """Trim a source sentence and collapse internal whitespace."""

    return _SPACE.sub(" ", str(value or "").strip())


def ensure_emotion_special_tokens(tokenizer) -> dict[str, int]:
    """Install and verify the production emotion markers as atomic tokens."""

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(EMOTION_SPECIAL_TOKENS)}
    )
    token_ids: dict[str, int] = {}
    for token in EMOTION_SPECIAL_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"情感 special token 不是原子 token: {token} -> {encoded}")
        token_id = int(encoded[0])
        if token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"情感 special token 被映射为 UNK: {token}")
        token_ids[token] = token_id
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("情感 special token ID 必须互不相同")
    return token_ids


def _sentence_id(sentence: Mapping[str, object]) -> str:
    value = sentence.get("sentenceId")
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError("context sentenceId 不能为空")
    return str(value)


def _section_id(sentence: Mapping[str, object]) -> str:
    value = sentence.get("sectionId")
    if value is None or str(value).strip() == "":
        raise ValueError("context sectionId 不能为空")
    return str(value)


def _line_index(sentence: Mapping[str, object]) -> int:
    value = sentence.get("lineIndex")
    if isinstance(value, bool):
        raise ValueError("context lineIndex 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("context lineIndex 必须是整数") from exc
    if result < 0:
        raise ValueError("context lineIndex 不能为负数")
    return result


def _sentence_type(sentence: Mapping[str, object]) -> str:
    value = str(sentence.get("sentenceType", sentence.get("type", ""))).strip()
    if value not in {"narration", "dialogue"}:
        raise ValueError(f"context sentenceType 无效: {value!r}")
    return value


def _normalized_sentence(sentence: Mapping[str, object]) -> dict[str, object]:
    text = normalize_context_sentence(sentence.get("text"))
    if not text:
        raise ValueError("context sentence text 不能为空")
    return {
        "sentenceId": sentence.get("sentenceId"),
        "sectionId": _section_id(sentence),
        "lineIndex": _line_index(sentence),
        "text": text,
        "sentenceType": _sentence_type(sentence),
    }


def _render_sentence(
    sentence: Mapping[str, object],
    *,
    target: bool,
    target_text_override: str | None = None,
) -> str:
    marker = DIALOGUE_TOKEN if _sentence_type(sentence) == "dialogue" else NARRATION_TOKEN
    text = (
        normalize_context_sentence(target_text_override)
        if target_text_override is not None
        else normalize_context_sentence(sentence.get("text"))
    )
    rendered = marker + text
    return TGT_OPEN + rendered + TGT_CLOSE if target else rendered


def render_target_context(
    sentences: Sequence[Mapping[str, object]],
    target_sentence_id: object,
    *,
    target_text_override: str | None = None,
) -> str:
    """Render selected sentences chronologically with line and target markers."""

    target_key = str(target_sentence_id)
    chunks: list[str] = []
    previous_line: int | None = None
    target_count = 0
    for sentence in sentences:
        current_line = _line_index(sentence)
        if previous_line is not None and current_line != previous_line:
            chunks.append(NL_TOKEN)
        is_target = _sentence_id(sentence) == target_key
        target_count += int(is_target)
        chunks.append(
            _render_sentence(
                sentence,
                target=is_target,
                target_text_override=target_text_override if is_target else None,
            )
        )
        previous_line = current_line
    if target_count != 1:
        raise ValueError(f"targetSentenceId 必须在 context 中唯一出现: {target_sentence_id}")
    return "".join(chunks)


@dataclass(frozen=True)
class TargetContextSelection:
    selected_positions: tuple[int, ...]
    selected_sentence_ids: tuple[str, ...]
    rendered_text: str
    input_token_count: int
    target_token_count: int
    previous_sentence_count: int
    next_sentence_included: bool
    context_limited: bool
    target_truncated: bool
    stop_reason: str


def _truncate_target_to_budget(
    target: Mapping[str, object],
    target_sentence_id: object,
    count_tokens: Callable[[str], int],
    max_length: int,
) -> tuple[str, int]:
    text = normalize_context_sentence(target.get("text"))
    low = 0
    high = len(text)
    best_rendered = render_target_context(
        [target], target_sentence_id, target_text_override=""
    )
    best_count = int(count_tokens(best_rendered))
    if best_count > max_length:
        raise ValueError("仅控制标记就超过 maxLength，无法保留 TGT ABI")
    while low <= high:
        middle = (low + high) // 2
        rendered = render_target_context(
            [target], target_sentence_id, target_text_override=text[:middle]
        )
        count = int(count_tokens(rendered))
        if count <= max_length:
            best_rendered = rendered
            best_count = count
            low = middle + 1
        else:
            high = middle - 1
    return best_rendered, best_count


def select_target_context(
    sentences: Sequence[Mapping[str, object]],
    target_sentence_id: object,
    count_tokens: Callable[[str], int],
    *,
    max_length: int = CONTEXT_MAX_LENGTH,
) -> TargetContextSelection:
    """Select a complete-sentence, section-local target context."""

    if max_length <= 0:
        raise ValueError("max_length 必须为正数")
    normalized = tuple(_normalized_sentence(sentence) for sentence in sentences)
    if not normalized:
        raise ValueError("context sentences 不能为空")
    section_ids = {_section_id(sentence) for sentence in normalized}
    if len(section_ids) != 1:
        raise ValueError("情感 context 不得跨 section")
    ids = [_sentence_id(sentence) for sentence in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("context sentenceId 不得重复")
    target_key = str(target_sentence_id)
    matches = [index for index, value in enumerate(ids) if value == target_key]
    if len(matches) != 1:
        raise ValueError("targetSentenceId 必须在 section 中唯一出现")
    target_pos = matches[0]
    target = normalized[target_pos]
    target_rendered = render_target_context([target], target_key)
    target_token_count = int(count_tokens(target_rendered))
    if target_token_count < 0:
        raise ValueError("token 计数不能为负数")
    if target_token_count > max_length:
        rendered, count = _truncate_target_to_budget(
            target, target_key, count_tokens, max_length
        )
        return TargetContextSelection(
            selected_positions=(target_pos,),
            selected_sentence_ids=(target_key,),
            rendered_text=rendered,
            input_token_count=count,
            target_token_count=target_token_count,
            previous_sentence_count=0,
            next_sentence_included=False,
            context_limited=target_pos > 0 or target_pos + 1 < len(normalized),
            target_truncated=True,
            stop_reason="target-over-max",
        )

    selected = {target_pos}
    context_limited = False
    previous_chain_open = True

    def try_add(position: int) -> bool:
        trial_positions = tuple(sorted((*selected, position)))
        trial_sentences = [normalized[index] for index in trial_positions]
        trial_text = render_target_context(trial_sentences, target_key)
        trial_count = int(count_tokens(trial_text))
        if trial_count < 0:
            raise ValueError("token 计数不能为负数")
        if trial_count > max_length:
            return False
        selected.add(position)
        return True

    if target_pos > 0 and not try_add(target_pos - 1):
        previous_chain_open = False
        context_limited = True

    next_pos = target_pos + 1
    next_is_eligible = (
        next_pos < len(normalized)
        and _line_index(normalized[next_pos]) == _line_index(target)
        and _section_id(normalized[next_pos]) == _section_id(target)
    )
    if next_is_eligible and not try_add(next_pos):
        context_limited = True

    if previous_chain_open:
        for position in range(target_pos - 2, -1, -1):
            if not try_add(position):
                context_limited = True
                break

    positions = tuple(sorted(selected))
    selected_sentences = [normalized[index] for index in positions]
    rendered = render_target_context(selected_sentences, target_key)
    input_count = int(count_tokens(rendered))
    if input_count > max_length:
        raise AssertionError("完整句选择器产生了超限输入")
    previous_count = sum(position < target_pos for position in positions)
    next_included = next_is_eligible and next_pos in selected
    if context_limited:
        stop_reason = "context-over-max"
    elif target_pos == 0:
        stop_reason = "section-start"
    else:
        stop_reason = "section-exhausted"
    return TargetContextSelection(
        selected_positions=positions,
        selected_sentence_ids=tuple(ids[index] for index in positions),
        rendered_text=rendered,
        input_token_count=input_count,
        target_token_count=target_token_count,
        previous_sentence_count=previous_count,
        next_sentence_included=next_included,
        context_limited=context_limited,
        target_truncated=False,
        stop_reason=stop_reason,
    )


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
    max_length: int = 256,
    sentence_limit: int = LEGACY_CONTEXT_SENTENCE_LIMIT,
    delimiter: str = LEGACY_CONTEXT_DELIMITER,
) -> ContextSelection:
    """Historical nearest-three pair policy retained for snapshot auditing."""

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
        return ContextSelection((), "", max_length, target_token_count, bool(candidates), True, "target-over-max")
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
    stop_reason = (
        "context-over-max"
        if stopped_for_budget
        else "window-start"
        if len(candidates) < sentence_limit
        else "context-limit"
    )
    return ContextSelection(
        selected,
        delimiter.join(selected),
        input_token_count,
        target_token_count,
        stopped_for_budget,
        False,
        stop_reason,
    )
