"""Export the reference-free reader model from full IndexTTS checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Mapping

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from indextts.voicepack.builder import model_fingerprint


RUNTIME_ABI = "indextts2.5-reader-runtime-v2"
MAX_CORE_MODEL_BYTES = int(3.2 * 1024**3)
CORE_FILES = ["gpt.safetensors", "s2mel.safetensors", "codec.safetensors", "bigvgan.safetensors"]
RUNTIME_PRECISION = {
    "gpt": "float32",
    "s2mel": "float32",
    "codec": "float32",
    "bigvgan": "float32",
    "float32Matmul": "ieee",
}
RUNTIME_CAPABILITIES = {
    "languages": ["zh"],
    "sampleRate": 22050,
    "requiresVoicePack": True,
    "emotionModes": ["base", "explicit"],
    "automaticEmotionIncluded": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_safetensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in sorted(tensors.items())}, temporary)
    os.replace(temporary, path)


def _load_torch(path: Path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def export_runtime_model(
    source_model_dir: str | Path,
    output_dir: str | Path,
    *,
    cfg_path: str | Path | None = None,
) -> dict:
    source = Path(source_model_dir).resolve()
    config = Path(cfg_path).resolve() if cfg_path else source / "config.yaml"
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录必须为空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(config)

    gpt_source = _load_torch(source / str(cfg.gpt_checkpoint))
    if "model" in gpt_source:
        gpt_source = gpt_source["model"]
    gpt_prefixes = (
        "gpt.",
        "text_embedding.",
        "lang_embedding.",
        "mel_embedding.",
        "mel_head.",
        "mel_pos_embedding.",
        "text_pos_embedding.",
        "final_norm.",
    )
    gpt_tensors = {
        key: value.float() if value.is_floating_point() else value
        for key, value in gpt_source.items()
        if key.startswith(gpt_prefixes)
    }
    _atomic_safetensors(gpt_tensors, output / "gpt.safetensors")
    del gpt_source, gpt_tensors

    s2mel_source = _load_torch(source / str(cfg.s2mel_checkpoint))["net"]
    s2mel_tensors = {}
    for module_name in ("cfm", "length_regulator"):
        for key, value in s2mel_source[module_name].items():
            if "input_pos" in key:
                continue
            s2mel_tensors[f"models.{module_name}.{key}"] = value.float() if value.is_floating_point() else value
    _atomic_safetensors(s2mel_tensors, output / "s2mel.safetensors")
    del s2mel_source, s2mel_tensors

    codec_source = _load_torch(source / "codec.pth")["model"]
    codec_tensors = {
        key: value.float() if value.is_floating_point() else value
        for key, value in codec_source.items()
        if key.startswith(("quantizer.", "decoder.", "up."))
    }
    _atomic_safetensors(codec_tensors, output / "codec.safetensors")
    del codec_source, codec_tensors

    bigvgan_dir = source / "hf_cache" / "bigvgan"
    bigvgan_source = _load_torch(bigvgan_dir / str(cfg.vocoder.name))["generator"]
    bigvgan_tensors = {
        key: value.float() if value.is_floating_point() else value
        for key, value in bigvgan_source.items()
    }
    _atomic_safetensors(bigvgan_tensors, output / "bigvgan.safetensors")
    del bigvgan_source, bigvgan_tensors

    shutil.copy2(config, output / "config.yaml")
    shutil.copy2(source / "multilingual_zh_ja_yue_char_del.tiktoken", output / "multilingual_zh_ja_yue_char_del.tiktoken")
    shutil.copy2(bigvgan_dir / "config.json", output / "bigvgan_config.json")
    repository_root = Path(__file__).resolve().parents[2]
    shutil.copy2(repository_root / "LICENSE", output / "LICENSE")
    shutil.copy2(repository_root / "LICENSE_ZH.txt", output / "LICENSE_ZH.txt")
    shutil.copy2(repository_root / "DISCLAIMER", output / "DISCLAIMER")

    core_bytes = sum((output / name).stat().st_size for name in CORE_FILES)
    if core_bytes > MAX_CORE_MODEL_BYTES:
        raise RuntimeError(f"质量优先核心模型超过 3.2 GiB: {core_bytes} bytes")
    files = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "runtime_model.json"
    }
    manifest = {
        "schemaVersion": 1,
        "runtimeAbi": RUNTIME_ABI,
        "indexTtsVersion": str(getattr(cfg, "version", "2.5")),
        "sourceModelFingerprint": model_fingerprint(source, config),
        "precision": RUNTIME_PRECISION,
        "capabilities": RUNTIME_CAPABILITIES,
        "coreModelBytes": core_bytes,
        "files": files,
        "license": "bilibili Model Use License",
    }
    manifest_path = output / "runtime_model.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_runtime_model(model_dir: str | Path) -> dict:
    root = Path(model_dir).resolve()
    manifest_path = root / "runtime_model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runtimeAbi") != RUNTIME_ABI:
        raise ValueError("运行时 ABI 不匹配")
    if manifest.get("capabilities") != RUNTIME_CAPABILITIES:
        raise ValueError("运行时能力声明不匹配")
    if manifest.get("precision") != RUNTIME_PRECISION:
        raise ValueError("运行时精度声明不匹配，质量优先运行时要求 FP32 GPT 和 IEEE FP32 矩阵计算")
    for name, spec in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != spec["bytes"] or _sha256_file(path) != spec["sha256"]:
            raise ValueError(f"运行时模型文件校验失败: {name}")
    actual = sum((root / name).stat().st_size for name in CORE_FILES)
    if actual != manifest["coreModelBytes"] or actual > MAX_CORE_MODEL_BYTES:
        raise ValueError("运行时核心模型大小无效")
    return manifest
