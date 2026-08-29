"""Pluggable text-to-emotion providers for reader synthesis."""

from __future__ import annotations

import json
import hashlib
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
        self.vector = normalize_emotion(vector, fallback_to_calm=False)

    def analyze(self, text: str) -> list[float]:
        self.warning = None
        return self.vector.copy()


class OnnxEmotionProvider(EmotionProvider):
    """FP32 MacBERT provider used for validation outside the Readest Rust host."""

    def __init__(self, model_path: str | Path, *, neutral_threshold: float | None = None):
        candidate = Path(model_path).resolve()
        self.model_dir = candidate.parent if candidate.is_file() else candidate
        self.neutral_threshold = None if neutral_threshold is None else float(neutral_threshold)
        self.max_length = 256
        self._session = None
        self._tokenizer = None
        self.warning = None

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("使用 ONNX 情感后端需要独立 emotion-onnx 工具环境") from exc
        from transformers import AutoTokenizer

        manifest_path = self.model_dir / "emotion_model.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("conditioningAbi") != "readest-emotion-v1"
            or manifest.get("labels") != EMOTION_NAMES
            or manifest.get("precision") != "fp32"
            or manifest.get("releaseStatus") != "approved"
        ):
            raise ValueError("ONNX 情感模型 manifest ABI、精度或发布状态无效")
        if self.neutral_threshold is None:
            self.neutral_threshold = float(manifest.get("neutralThreshold", 0.15))
        self.max_length = int(manifest.get("maxLength", 256))
        if not 1 <= self.max_length <= 512:
            raise ValueError("ONNX 情感模型 manifest 的 maxLength 必须位于 [1, 512]")
        for name, metadata in manifest.get("files", {}).items():
            if (
                Path(name).name != name
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or ":" in name
            ):
                raise ValueError(f"ONNX 情感模型 manifest 含不安全路径: {name}")
            path = self.model_dir / name
            if path.stat().st_size != int(metadata["bytes"]):
                raise ValueError(f"ONNX 情感模型文件大小不匹配: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError(f"ONNX 情感模型文件哈希不匹配: {name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, use_fast=True
        )
        self._session = ort.InferenceSession(
            str(self.model_dir / "emotion.onnx"), providers=["CPUExecutionProvider"]
        )

    def analyze_context(
        self,
        previous_text: str,
        text: str,
        sentence_type: str = "narration",
    ) -> list[float]:
        self._load()
        import numpy as np

        current = ("[对白]" if sentence_type == "dialogue" else "[旁白]") + text.strip()
        batch = self._tokenizer(
            [previous_text.strip()],
            [current],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="np",
        )
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = np.zeros_like(batch["input_ids"])
        vector, intensity = self._session.run(
            ["emotion_vector", "total_intensity"],
            {
                name: np.asarray(batch[name], dtype=np.int64)
                for name in ("input_ids", "attention_mask", "token_type_ids")
            },
        )
        if vector.shape != (1, 8) or intensity.shape != (1, 1):
            raise ValueError(
                f"ONNX 情感模型输出形状无效: vector={vector.shape}, intensity={intensity.shape}"
            )
        self.warning = None
        if float(intensity[0, 0]) < float(self.neutral_threshold):
            return [0.0] * 8
        return [max(0.0, min(1.0, float(value))) for value in vector[0]]

    def analyze(self, text: str) -> list[float]:
        return self.analyze_context("", text, "narration")


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
