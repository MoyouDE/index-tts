"""Chinese-only text preparation for the reader runtime."""

from __future__ import annotations

import re


PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
PROTECTED_PATTERN = re.compile(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")


def apply_pronunciation_annotations(text: str) -> str:
    def replace(match):
        word = match.group(1)
        pronunciation = match.group(2).upper()
        token = "SPECIAL_TOKEN_2" if re.search(r"[\u4e00-\u9fff]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pronunciation}<|{token}|>"

    return PRONUNCIATION_PATTERN.sub(replace, text)


def split_text_by_punctuation(text: str, max_chars: int = 40) -> list[str]:
    parts = re.split(r"(?<=[，。！？、；：,\.!\?;:\n])", text)
    segments: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if current and len(current) + len(part) > max_chars:
            segments.append(current)
            current = part
        elif len(part) > max_chars and not current:
            for offset in range(0, len(part), max_chars):
                chunk = part[offset:offset + max_chars]
                if len(chunk) == max_chars:
                    segments.append(chunk)
                else:
                    current = chunk
        else:
            current += part
    if current:
        segments.append(current)
    return segments or [text]


def split_text_by_tokens(text: str, tokenizer, capacity: int, lang_prefix: str, max_tokens: int = 120) -> list[str]:
    token_len = lambda value: len(tokenizer.encode(value, allowed_special="all"))
    budget = max(1, min(max_tokens, capacity - 2) - token_len(lang_prefix))
    if token_len(text) <= budget:
        return [text]
    pieces: list[tuple[str, bool]] = []
    position = 0
    for match in PROTECTED_PATTERN.finditer(text):
        if match.start() > position:
            pieces.append((text[position:match.start()], False))
        pieces.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))

    chunks: list[str] = []
    for piece, protected in pieces:
        if protected:
            chunks.append(piece)
            continue
        for part in re.split(r"(?<=[，。！？、；：,\.!\?;:\n])", piece):
            if not part:
                continue
            current = ""
            for character in part:
                if current and token_len(current + character) > budget:
                    chunks.append(current)
                    current = character
                else:
                    current += character
            if current:
                chunks.append(current)

    segments: list[str] = []
    current = ""
    for chunk in chunks:
        if current and token_len(current + chunk) > budget:
            segments.append(current)
            current = chunk
        else:
            current += chunk
    if current:
        segments.append(current)
    return segments or [text]
