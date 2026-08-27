"""Reference-free IndexTTS synthesis engine for desktop reader sidecars."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.runtime.codec import SemanticCodecDecoder
from indextts.runtime.emotion import CALM_VECTOR, EmotionProvider, normalize_emotion
from indextts.runtime.model_export import verify_runtime_model
from indextts.runtime.text import apply_pronunciation_annotations, split_text_by_punctuation, split_text_by_tokens
from indextts.runtime.tokenizer import ReaderTokenizer
from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN, load_hparams_from_json
from indextts.s2mel.modules.commons import MyModel
from indextts.utils.front import TextNormalizer
from indextts.utils.languages import lang_to_token
from indextts.voicepack import VoicePackError, load_voicepack


class SynthesisCancelled(RuntimeError):
    pass


def _write_pcm16(path: Path, pcm: torch.Tensor, sample_rate: int) -> None:
    array = pcm.detach().cpu().clamp(-32767, 32767).to(torch.int16).numpy()
    if array.ndim == 2:
        array = array.squeeze(0)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(array.astype("<i2", copy=False).tobytes())


class ReaderRuntime:
    sample_rate = 22050

    def __init__(
        self,
        model_dir: str | Path,
        voice_dirs: Sequence[str | Path],
        emotion_provider: EmotionProvider | None,
        device: str = "cuda:0",
        cache_dir: str | Path = "outputs/runtime-cache",
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.voice_dirs = [Path(directory).resolve() for directory in voice_dirs]
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.emotion_provider = emotion_provider
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ReaderRuntime v1 需要 NVIDIA CUDA GPU")
        self.manifest = verify_runtime_model(self.model_dir)
        self.source_fingerprint = self.manifest["sourceModelFingerprint"]
        self.cfg = OmegaConf.load(self.model_dir / "config.yaml")
        self.dtype = torch.bfloat16
        self.stop_mel_token = int(self.cfg.gpt.stop_mel_token)
        self._synthesis_lock = threading.Lock()
        self._voices: dict[str, Any] = {}
        self._load_models()
        self._load_text_frontend()
        self.reload_voices()

    def _load_models(self) -> None:
        self.gpt = UnifiedVoice(
            **self.cfg.gpt,
            use_accel=False,
            spk_cond_mode="campplus",
            precomputed_conditioning=True,
        )
        self.gpt.load_state_dict(load_file(self.model_dir / "gpt.safetensors"), strict=True)
        self.gpt = self.gpt.to(self.device).eval().bfloat16()
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=True)

        self.semantic_codec = SemanticCodecDecoder(**self.cfg.semantic_codec, cfg=self.cfg.semantic_codec)
        self.semantic_codec.load_state_dict(load_file(self.model_dir / "codec.safetensors"), strict=True)
        self.semantic_codec = self.semantic_codec.to(self.device).eval()

        self.s2mel = MyModel(self.cfg.s2mel)
        missing, unexpected = self.s2mel.load_state_dict(
            load_file(self.model_dir / "s2mel.safetensors"), strict=False
        )
        allowed_missing = {"models.cfm.estimator.input_pos"}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"s2mel 裁剪权重不兼容: missing={missing}, unexpected={unexpected}"
            )
        self.s2mel = self.s2mel.to(self.device).eval()
        self.s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        hparams = load_hparams_from_json(self.model_dir / "bigvgan_config.json")
        self.bigvgan = BigVGAN(hparams, use_cuda_kernel=False)
        self.bigvgan.load_state_dict(load_file(self.model_dir / "bigvgan.safetensors"), strict=True)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

    def _load_text_frontend(self) -> None:
        self.tokenizer = ReaderTokenizer(self.model_dir)
        self.text_process = TextNormalizer(enable_glossary=False)
        self.text_process.load()

    def reload_voices(self) -> list[dict[str, Any]]:
        voices: dict[str, Any] = {}
        for directory in self.voice_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.ivp")):
                pack = load_voicepack(path, expected_model_fingerprint=self.source_fingerprint)
                voice_id = pack.voice_id
                if voice_id in voices:
                    raise VoicePackError(f"发现重复 voiceId: {voice_id}")
                voices[voice_id] = pack
        self._voices = voices
        return self.list_voices()

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {
                "voiceId": voice_id,
                "displayName": pack.manifest["displayName"],
                "gender": pack.manifest["gender"],
                "language": pack.manifest["language"],
                "path": str(pack.path),
            }
            for voice_id, pack in sorted(self._voices.items())
        ]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device),
            "voiceCount": len(self._voices),
            "runtimeAbi": self.manifest["runtimeAbi"],
        }

    def _cancel_boundary(self, cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise SynthesisCancelled("合成已取消")

    def _emotion_vector(self, text: str, emotion: str | Sequence[float]) -> tuple[list[float], list[str]]:
        warnings: list[str] = []
        if emotion == "auto":
            if self.emotion_provider is None:
                warnings.append("未配置自动情感后端，已回退 calm")
                return CALM_VECTOR.copy(), warnings
            try:
                vector = self.emotion_provider.analyze(text)
                warning = getattr(self.emotion_provider, "warning", None)
                if warning:
                    warnings.append(warning)
                return normalize_emotion(vector, apply_bias=False), warnings
            except Exception as exc:
                warnings.append(f"自动情感分析失败，已回退 calm: {exc}")
                return CALM_VECTOR.copy(), warnings
        if isinstance(emotion, str):
            raise ValueError("emotion 只能是 auto 或 8 维显式向量")
        return normalize_emotion(emotion), warnings

    def _prepare_segments(self, text: str) -> list[torch.Tensor]:
        text = self.text_process.clean_pattern.sub(lambda match: self.text_process.char_rep_map[match.group()], text)
        text = self.text_process.normalize(text).lower()
        text = apply_pronunciation_annotations(text)
        text = re.sub(r"<\|([^|]+)\|>", lambda match: f"<|{match.group(1).upper()}|>", text)
        prefix = "<|zh|> "
        capacity = self.gpt.text_pos_embedding.emb.num_embeddings
        raw_segments = []
        for low_vram_segment in split_text_by_punctuation(text, max_chars=40):
            raw_segments.extend(
                split_text_by_tokens(low_vram_segment, self.tokenizer, capacity, prefix, max_tokens=120)
            )
        tokens = []
        for segment in raw_segments:
            encoded = self.tokenizer.encode(prefix + segment, allowed_special="all")
            tensor = torch.tensor(encoded, dtype=torch.int32, device=self.device).unsqueeze(0)
            tokens.append(F.pad(tensor, (0, 1), value=1))
        return tokens

    def _voice_condition(self, voice_id: str, vector: Sequence[float]):
        if voice_id not in self._voices:
            raise KeyError(f"未知音色: {voice_id}")
        tensors = self._voices[voice_id].tensors
        speaker = tensors["speaker_latent"].to(self.device, dtype=self.dtype)
        base = tensors["base_emotion"].to(self.device, dtype=self.dtype)
        # Match the full inference path exactly: selected emotion bases and
        # weights are FP32, while the encoded base emotion and speaker projection
        # are BF16. The promotion to FP32 here prevents late autoregressive drift.
        basis = tensors["emotion_basis"].to(self.device, dtype=torch.float32)
        weights = torch.tensor(vector, device=self.device, dtype=torch.float32)
        emotion = torch.sum(weights.unsqueeze(1) * basis, dim=0, keepdim=True)
        emotion = emotion + (1 - weights.sum()) * base
        first = (speaker + emotion).unsqueeze(1)
        zeros = torch.zeros((1, 2, first.shape[-1]), device=self.device, dtype=first.dtype)
        return (
            torch.cat([first, zeros], dim=1),
            tensors["prompt_condition"].to(self.device, dtype=torch.float32),
            tensors["ref_mel"].to(self.device, dtype=torch.float32),
            tensors["speaker_style"].to(self.device, dtype=torch.float32),
        )

    def synthesize(
        self,
        text: str,
        voice_id: str,
        emotion: str | Sequence[float] = "auto",
        duration_factor: float = 1.0,
        *,
        _cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text 不能为空")
        if not 0.5 <= float(duration_factor) <= 2.0:
            raise ValueError("duration_factor 必须位于 [0.5, 2.0]")
        with self._synthesis_lock:
            started = time.perf_counter()
            timings: dict[str, float] = {}
            temporary = self.cache_dir / f".{uuid.uuid4().hex}.wav.part"
            try:
                self._cancel_boundary(_cancelled)
                stage = time.perf_counter()
                vector, warnings = self._emotion_vector(text, emotion)
                timings["emotionMs"] = (time.perf_counter() - stage) * 1000
                self._cancel_boundary(_cancelled)

                stage = time.perf_counter()
                segments = self._prepare_segments(text)
                conditional, prompt, ref_mel, style = self._voice_condition(voice_id, vector)
                timings["prepareMs"] = (time.perf_counter() - stage) * 1000
                wavs = []
                gpt_seconds = acoustic_seconds = vocoder_seconds = 0.0
                language = torch.tensor([lang_to_token("zh")], dtype=torch.long, device=self.device)
                for index, text_tokens in enumerate(segments):
                    self._cancel_boundary(_cancelled)
                    stage = time.perf_counter()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
                        codes = self.gpt.inference_speech_from_conditioning(
                            conditional,
                            text_tokens,
                            language,
                            do_sample=False,
                            num_beams=3,
                            repetition_penalty=10.0,
                            length_penalty=0.0,
                            max_generate_length=1500,
                            num_return_sequences=1,
                        )
                    if not isinstance(codes, torch.Tensor):
                        codes = codes.sequences
                    gpt_seconds += time.perf_counter() - stage
                    self._cancel_boundary(_cancelled)

                    if self.stop_mel_token in codes[0]:
                        stop = (codes[0] == self.stop_mel_token).nonzero(as_tuple=False)[0, 0].item()
                        codes = codes[:, :stop]
                    if codes.shape[1] == 0:
                        raise RuntimeError(f"第 {index + 1} 段未生成有效语义 token")
                    stage = time.perf_counter()
                    with torch.no_grad():
                        semantic = self.semantic_codec.decode(codes)
                        target_lengths = torch.tensor(
                            [int(semantic.shape[1] * 1.72 * float(duration_factor))],
                            dtype=torch.long,
                            device=self.device,
                        )
                        condition = self.s2mel.models["length_regulator"](
                            semantic, ylens=target_lengths, n_quantizers=3, f0=None
                        )[0]
                        combined = torch.cat([prompt, condition], dim=1)
                        mel = self.s2mel.models["cfm"].inference(
                            combined,
                            torch.tensor([combined.size(1)], dtype=torch.long, device=self.device),
                            ref_mel,
                            style,
                            None,
                            25,
                            inference_cfg_rate=0.7,
                        )
                        mel = mel[:, :, ref_mel.size(-1):]
                    acoustic_seconds += time.perf_counter() - stage
                    self._cancel_boundary(_cancelled)

                    stage = time.perf_counter()
                    with torch.no_grad():
                        wav = self.bigvgan(mel.float()).squeeze().unsqueeze(0)
                    vocoder_seconds += time.perf_counter() - stage
                    wavs.append(torch.clamp(32767 * wav, -32767.0, 32767.0).cpu())
                    self._cancel_boundary(_cancelled)

                silence = torch.zeros(1, int(self.sample_rate * 0.2))
                parts = []
                for index, wav in enumerate(wavs):
                    parts.append(wav)
                    if index + 1 < len(wavs):
                        parts.append(silence)
                pcm = torch.cat(parts, dim=1)
                output = self.cache_dir / f"tts-{uuid.uuid4().hex}.wav"
                _write_pcm16(temporary, pcm, self.sample_rate)
                os.replace(temporary, output)
                duration_ms = round(pcm.shape[-1] * 1000 / self.sample_rate)
                timings.update(
                    {
                        "gptMs": gpt_seconds * 1000,
                        "acousticMs": acoustic_seconds * 1000,
                        "vocoderMs": vocoder_seconds * 1000,
                        "totalMs": (time.perf_counter() - started) * 1000,
                    }
                )
                return {
                    "audioPath": str(output),
                    "sampleRate": self.sample_rate,
                    "durationMs": duration_ms,
                    "emotionVector": vector,
                    "timings": {key: round(value, 2) for key, value in timings.items()},
                    "warnings": warnings,
                }
            finally:
                if temporary.exists():
                    temporary.unlink()
