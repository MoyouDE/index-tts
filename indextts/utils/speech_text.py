"""Conservative cleanup of decorative text before speech normalization.

Keep words, pronunciation annotations, numbers and arithmetic intact. Emoji
are visual decorations, not acoustic control tokens; conversational tildes
are pauses, while numeric ranges still go through the number normalizer.
No optional packages or model downloads are needed.
"""

import re


_KEYCAP = re.compile(r"([0-9#*])\ufe0f?\u20e3")
_PICTOGRAPH = (
    r"[\U0001f000-\U0001faff\u2600-\u27bf\u231a\u231b"
    r"\u23e9-\u23f3\u23f8-\u23fa\u2b50\u2b55\u25a0-\u25a1\u25b2\u25bc\u25c6-\u25c7]"
)
_EMOJI = re.compile(
    r"(?:" + _PICTOGRAPH + r"[\ufe0e\ufe0f\U0001f3fb-\U0001f3ff\U000e0020-\U000e007f]*"
    + r"(?:\u200d" + _PICTOGRAPH
    + r"[\ufe0e\ufe0f\U0001f3fb-\U0001f3ff]*)*)+"
)
_WAVES = re.compile(r"[~～〜]+")
_PUNCTUATION = ",.?!;:，。？！；：…"
_CLOSERS = "\"'”’」』）》】)]}"


def has_spoken_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


def sanitize_speech_text(text: str) -> str:
    """Return idempotent, readable speech text; never invent spoken words."""
    # Markdown escapes attached to symbols must not leave a spoken backslash.
    text = re.sub(r"\\(?=[^\w\s])", "", text)
    text = _KEYCAP.sub(r"\1", text)
    emoji_at_end = bool(re.search(_EMOJI.pattern + r"[\s" + re.escape(_CLOSERS) + r"]*$", text))
    def remove_emoji(match):
        # Only join CJK characters that were directly separated by decoration;
        # preserve spaces the author actually typed, and separate English words.
        left = text[match.start() - 1:match.start()] if match.start() else ""
        right = text[match.end():match.end() + 1]
        return "" if left and right and all("\u3400" <= c <= "\u9fff" for c in left + right) else " "

    text = _EMOJI.sub(remove_emoji, text)
    text = text.replace("\ufe0e", "").replace("\ufe0f", "")
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")

    def replace_wave(match):
        before = text[:match.start()].rstrip()
        after = text[match.end():].lstrip()
        if before and after and before[-1].isdigit() and after[0].isdigit():
            return "-"
        if not before or before[-1] in _PUNCTUATION or (after and after[0] in _PUNCTUATION):
            return ""
        return "." if not after.strip(_CLOSERS + " ") else ","

    text = _WAVES.sub(replace_wave, text)
    text = re.sub(r"[!?！？]{2,}", lambda m: "?" if any(c in m[0] for c in "?？") else "!", text)
    text = re.sub(r"\.{3,}|…{2,}", "…", text)
    text = re.sub(r"[,，]{2,}", ",", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r" +([,.?!;:，。？！；：…])", r"\1", text)
    if emoji_at_end and text and text[-1].isalnum():
        text += "."
    return text if has_spoken_content(text) else ""
