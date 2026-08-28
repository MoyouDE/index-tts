"""Pluggable text-to-emotion providers for reader synthesis."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


EMOTION_NAMES = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]
EMOTION_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
CALM_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]


def normalize_emotion(
    vector: Sequence[float],
    *,
    apply_bias: bool = True,
    fallback_to_calm: bool = True,
) -> list[float]:
    if len(vector) != 8:
        raise ValueError("情感向量必须正好包含 8 个数值")
    normalized = []
    for index, value in enumerate(vector):
        number = float(value)
        if not 0.0 <= number <= 1.2:
            raise ValueError(f"情感分量 {EMOTION_NAMES[index]} 超出 [0, 1.2]")
        normalized.append(number * EMOTION_BIAS[index] if apply_bias else number)
    total = sum(normalized)
    if total > 0.8:
        normalized = [value * 0.8 / total for value in normalized]
    if fallback_to_calm and not any(normalized):
        return CALM_VECTOR.copy()
    return normalized


class EmotionProvider(ABC):
    warning: str | None = None

    @abstractmethod
    def analyze(self, text: str) -> list[float]:
        """Return [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]."""


class ExplicitEmotionProvider(EmotionProvider):
    def __init__(self, vector: Sequence[float]):
        self.vector = normalize_emotion(vector)

    def analyze(self, text: str) -> list[float]:
        self.warning = None
        return self.vector.copy()


class OnnxEmotionProvider(EmotionProvider):
    """Reserved ABI for a future compact classifier; no model is distributed yet."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def analyze(self, text: str) -> list[float]:
        raise NotImplementedError("本版本尚未分发 ONNX 情感分类器")


class QwenEmotionProvider(EmotionProvider):
    _CN_KEYS = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
    _ALIASES = dict(zip(_CN_KEYS, EMOTION_NAMES))
    _MELANCHOLIC_WORDS = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"}

    def __init__(self, model_dir: str | Path, *, max_new_tokens: int = 128, timeout_seconds: float = 30.0):
        self.model_dir = Path(model_dir).resolve()
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self._tokenizer = None
        self._model = None
        self.warning = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            local_files_only=True,
        )
        self._model.eval()
        if any(parameter.device.type != "cpu" for parameter in self._model.parameters()):
            raise RuntimeError("QwenEmotionProvider 必须完全位于 CPU")

    def _decode_vector(self, response: str, source_text: str) -> list[float]:
        try:
            content = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Qwen 返回了无效 JSON") from exc
        if isinstance(content, str):
            content = {content: 1.0}
        if not isinstance(content, dict):
            raise ValueError("Qwen 返回值不是 JSON 对象")
        vector = [0.0] * 8
        for key, value in content.items():
            normalized_key = self._ALIASES.get(str(key), str(key).lower())
            if normalized_key in EMOTION_NAMES:
                vector[EMOTION_NAMES.index(normalized_key)] = max(0.0, min(1.2, float(value)))
        lowered = source_text.lower()
        if any(word in lowered for word in self._MELANCHOLIC_WORDS):
            vector[2], vector[5] = vector[5], vector[2]
        return normalize_emotion(vector)

    def analyze(self, text: str) -> list[float]:
        self.warning = None
        try:
            self._load()
            messages = [
                {"role": "system", "content": "文本情感分类。只输出包含八类情感分数的 JSON。"},
                {"role": "user", "content": text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self._tokenizer([prompt], return_tensors="pt")
            started = time.perf_counter()
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                max_time=self.timeout_seconds,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            if time.perf_counter() - started >= self.timeout_seconds:
                raise TimeoutError(f"Qwen 情感分析超过 {self.timeout_seconds:g} 秒")
            response_ids = generated[0][inputs.input_ids.shape[1]:]
            response = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            return self._decode_vector(response, text)
        except Exception as exc:
            self.warning = f"自动情感分析失败，已回退 calm: {exc}"
            return CALM_VECTOR.copy()
