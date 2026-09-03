"""FP32 ONNX export, manifest generation and numerical verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .dataset import format_current_text
from .context_policy import (
    CONTEXT_DELIMITER,
    CONTEXT_MAX_LENGTH,
    CONTEXT_POLICY,
    CONTEXT_SENTENCE_LIMIT,
)
from .model import EmotionOnnxWrapper, load_checkpoint
from .release import validate_release_approval
from .schema import EMOTION_NAMES


ONNX_FILENAME = "emotion.onnx"
MANIFEST_FILENAME = "emotion_model.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    release_approval: dict[str, object] | None = None,
    version: str = "1.0.0",
    neutral_threshold: float = 0.15,
) -> dict[str, object]:
    if not version.strip():
        raise ValueError("version 不能为空")
    if not 0.0 <= float(neutral_threshold) <= 1.0:
        raise ValueError("neutral_threshold 必须位于 [0, 1]")
    if release_approval is not None:
        release_approval = validate_release_approval(release_approval)
    source = Path(checkpoint_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_checkpoint(source)
    model.float().eval()
    wrapper = EmotionOnnxWrapper(model).eval()
    sample = tokenizer(
        ["夜色渐深。"],
        [format_current_text("他终于回来了。", "narration")],
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    if "token_type_ids" not in sample:
        sample["token_type_ids"] = torch.zeros_like(sample["input_ids"])
    onnx_path = target / ONNX_FILENAME
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["emotion_vector", "total_intensity"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "emotion_vector": {0: "batch"},
            "total_intensity": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    tokenizer_files = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
            tokenizer_files.append(name)
    notices = """# Third-party notices\n\n- Base model: hfl/chinese-macbert-base (Apache-2.0)\n- Public supervised data: BRIGHTER emotion intensities (CC BY 4.0)\n- The release model must also carry attribution for every proprietary-authorized training work.\n- No Qwen pseudo-label is permitted in the release checkpoint.\n"""
    (target / "THIRD_PARTY_NOTICES.md").write_text(notices, encoding="utf-8")
    files = [ONNX_FILENAME, *tokenizer_files, "THIRD_PARTY_NOTICES.md"]
    manifest = {
        "schemaVersion": 1,
        "package": "readest-macbert-emotion",
        "version": version,
        "conditioningAbi": "readest-emotion-v2",
        "contextPolicy": CONTEXT_POLICY,
        "contextSentenceLimit": CONTEXT_SENTENCE_LIMIT,
        "contextDelimiter": CONTEXT_DELIMITER,
        "language": "zh",
        "precision": "fp32",
        "maxLength": CONTEXT_MAX_LENGTH,
        "neutralThreshold": float(neutral_threshold),
        "labels": list(EMOTION_NAMES),
        "inputs": ["input_ids", "attention_mask", "token_type_ids"],
        "outputs": ["emotion_vector", "total_intensity"],
        "baseModel": "hfl/chinese-macbert-base",
        "baseModelLicense": "Apache-2.0",
        "sourceCheckpointSha256": sha256_file(source / "model.safetensors"),
        "trainingDataPolicy": "commercial-license-allowlist; no-qwen-pseudo-labels",
        "releaseStatus": "approved" if release_approval is not None else "candidate-unvalidated",
        "releaseApproval": release_approval,
        "files": {
            name: {"sha256": sha256_file(target / name), "bytes": (target / name).stat().st_size}
            for name in files
        },
    }
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_onnx(
    checkpoint_dir: str | Path,
    model_dir: str | Path,
    *,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("验证 ONNX 需要安装 emotion_train extra") from exc
    source = Path(checkpoint_dir)
    target = Path(model_dir)
    model, tokenizer = load_checkpoint(source)
    model.float().eval()
    previous = ["她握紧了拳头。", "雨停了。"]
    current = [
        format_current_text("你怎么敢这样骗我！", "dialogue"),
        format_current_text("湖面重新恢复了宁静。", "narration"),
    ]
    batch = tokenizer(
        previous,
        current,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
    with torch.inference_mode():
        expected = model(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )
    session = ort.InferenceSession(str(target / ONNX_FILENAME), providers=["CPUExecutionProvider"])
    actual_vector, actual_intensity = session.run(
        ["emotion_vector", "total_intensity"],
        {name: batch[name].numpy().astype(np.int64) for name in ("input_ids", "attention_mask", "token_type_ids")},
    )
    vector_error = float(np.max(np.abs(actual_vector - expected.emotion_vector.numpy())))
    intensity_error = float(np.max(np.abs(actual_intensity - expected.intensity.numpy())))
    if max(vector_error, intensity_error) > tolerance:
        raise RuntimeError(
            f"ONNX 数值偏差超限: vector={vector_error:g}, intensity={intensity_error:g}"
        )
    manifest = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        actual_hash = sha256_file(target / name)
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(f"文件哈希不匹配: {name}")
    return {
        "ok": True,
        "vectorMaxAbsError": vector_error,
        "intensityMaxAbsError": intensity_error,
        "modelBytes": (target / ONNX_FILENAME).stat().st_size,
    }
