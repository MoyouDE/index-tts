"""Minimal IndexTTS tokenizer without OpenAI Whisper's audio package imports."""

from __future__ import annotations

import base64
from pathlib import Path

import tiktoken

from indextts.utils.languages import LANGUAGE_CODES


_AUDIO_EVENTS = [
    "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause"
]
_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]
_TTS_TOKENS = ["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"] + [
    f"TTS/SP{index:02d}" for index in range(1, 14)
]


class ReaderTokenizer:
    def __init__(self, model_dir: str | Path):
        vocab_path = Path(model_dir) / "multilingual_zh_ja_yue_char_del.tiktoken"
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line)
        }
        special_names = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{language}|>" for language in LANGUAGE_CODES[:99]],
            *[f"<|{event}|>" for event in _AUDIO_EVENTS],
            *[f"<|{emotion}|>" for emotion in _EMOTIONS],
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            *[f"<|SPECIAL_TOKEN_{index}|>" for index in range(1, 31)],
            *[f"<|{token}|>" for token in _TTS_TOKENS],
            *[f"<|{index * 0.02:.2f}|>" for index in range(1501)],
        ]
        special_tokens = {name: len(ranks) + index for index, name in enumerate(special_names)}
        self.encoding = tiktoken.Encoding(
            name=vocab_path.name,
            explicit_n_vocab=len(ranks) + len(special_tokens),
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )

    def encode(self, text: str, **kwargs) -> list[int]:
        return self.encoding.encode(text, **kwargs)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)
