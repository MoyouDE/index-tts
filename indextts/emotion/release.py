"""Model-quality evidence envelopes and legacy benchmark support."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .metrics import emotion_metrics
from .schema import EMOTION_NAMES, EmotionExample


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def benchmark_qwen_annotations(
    examples: Sequence[EmotionExample],
    annotation_paths: Sequence[str | Path],
    *,
    neutral_adjust_calm: bool = True,
) -> dict[str, object]:
    """Evaluate previously generated Qwen vectors on an exact held-out snapshot."""

    if not annotation_paths:
        raise ValueError("至少需要一个 Qwen 标注文件")
    annotations: dict[str, Mapping[str, object]] = {}
    source_hashes: dict[str, str] = {}
    for raw_path in annotation_paths:
        path = Path(raw_path)
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Qwen 标注必须是 JSON 对象: {path}:{line_number}")
                row_id = str(row.get("id", "")).strip()
                if not row_id:
                    raise ValueError(f"Qwen 标注缺少 id: {path}:{line_number}")
                if row_id in annotations:
                    raise ValueError(f"Qwen 标注 id 重复: {row_id}")
                annotations[row_id] = row

    predictions: list[list[float]] = []
    missing: list[str] = []
    for example in examples:
        row = annotations.get(example.example_id)
        if row is None:
            missing.append(example.example_id)
            continue
        expected_text_hash = row.get("textSha256")
        if expected_text_hash and expected_text_hash != _text_sha256(example.text):
            raise ValueError(f"Qwen 标注 textSha256 不匹配: {example.example_id}")
        expected_previous_hash = row.get("previousTextSha256")
        if expected_previous_hash and expected_previous_hash != _text_sha256(
            example.previous_text
        ):
            raise ValueError(f"Qwen 标注 previousTextSha256 不匹配: {example.example_id}")
        emotions = row.get("emotions")
        if not isinstance(emotions, Mapping) or set(emotions) != set(EMOTION_NAMES):
            raise ValueError(f"Qwen 标注八维键集合无效: {example.example_id}")
        vector = [max(0.0, min(1.0, float(emotions[name]))) for name in EMOTION_NAMES]
        if neutral_adjust_calm:
            # The bundled IndexTTS Qwen model names its neutral/default basis
            # "natural".  Dataset calm is reserved for active composure, so
            # direct classifier comparison must remove that runtime-only basis.
            vector[EMOTION_NAMES.index("calm")] = 0.0
        predictions.append(vector)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Qwen 标注未覆盖测试集 {len(missing)} 条，示例: {preview}")

    labels = np.asarray([example.labels for example in examples], dtype=np.float32)
    mask = np.asarray([example.label_mask for example in examples], dtype=np.float32)
    intensities = np.asarray([example.intensity for example in examples], dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    predicted_intensities = prediction_array.max(axis=1)
    metrics = emotion_metrics(
        prediction_array,
        labels,
        mask,
        intensities,
        predicted_intensities,
    )
    metrics.update(
        {
            "backend": "qwen0.6bemo4-merge-precomputed",
            "semanticPolicy": "natural-as-base" if neutral_adjust_calm else "raw-calm",
            "annotationSources": source_hashes,
            "warningCount": 0,
            "latencyAvailable": False,
        }
    )
    return metrics


def benchmark_qwen(
    examples: Sequence[EmotionExample],
    model_dir: str | Path,
) -> dict[str, object]:
    from indextts.runtime.emotion import QwenEmotionProvider

    provider = QwenEmotionProvider(model_dir)
    predictions: list[list[float]] = []
    warnings: list[str] = []
    latencies: list[float] = []
    for example in examples:
        prompt = (
            f"上文：{example.previous_text}\n"
            f"类型：{'对白' if example.sentence_type == 'dialogue' else '旁白'}\n"
            f"当前：{example.text}"
        )
        started = time.perf_counter()
        predictions.append(provider.analyze(prompt))
        latencies.append((time.perf_counter() - started) * 1000)
        if provider.warning:
            warnings.append(f"{example.example_id}: {provider.warning}")
    labels = np.asarray([example.labels for example in examples], dtype=np.float32)
    mask = np.asarray([example.label_mask for example in examples], dtype=np.float32)
    intensities = np.asarray([[example.intensity] for example in examples], dtype=np.float32)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    predicted_intensities = prediction_array.sum(axis=1, keepdims=True).clip(0, 1)
    metrics = emotion_metrics(
        prediction_array,
        labels,
        mask,
        intensities,
        predicted_intensities,
    )
    metrics.update(
        {
            "backend": "qwen0.6bemo4-merge",
            "warningCount": len(warnings),
            "warnings": warnings,
            "meanLatencyMs": float(np.mean(latencies)),
            "p95LatencyMs": float(np.percentile(latencies, 95)),
        }
    )
    return metrics


def _read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"发布门槛输入必须是 JSON 对象: {source}")
    return value


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_release_approval(approval: object) -> dict[str, object]:
    """Validate the automated model-quality envelope used by packaged exports."""
    if not isinstance(approval, dict) or approval.get("schemaVersion") != 2:
        raise ValueError("发布质量文件 schemaVersion 必须为 2")
    if approval.get("qualityGatePassed") is not True:
        raise ValueError("发布质量文件未通过自动验收")
    if approval.get("selectionPolicy") != "experiment-comparison-and-user-selection":
        raise ValueError("发布质量文件的 selectionPolicy 无效")
    if not str(approval.get("selectedCandidate", "")).strip():
        raise ValueError("发布质量文件缺少 selectedCandidate")
    input_hashes = approval.get("inputSha256")
    expected_inputs = {
        "trainingReport",
        "devMetrics",
        "testMetrics",
        "experimentAudit",
    }
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_inputs:
        raise ValueError("发布质量文件的 inputSha256 证据集合不完整")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
        for value in input_hashes.values()
    ):
        raise ValueError("发布质量文件包含无效的 SHA-256")
    return approval


def build_release_approval(
    *,
    training_report_path: str | Path,
    dev_metrics_path: str | Path,
    test_metrics_path: str | Path,
    experiment_audit_path: str | Path,
) -> dict[str, object]:
    training = _read_json(training_report_path)
    dev = _read_json(dev_metrics_path)
    test = _read_json(test_metrics_path)
    audit = _read_json(experiment_audit_path)

    objective = training.get("trainingObjective", training.get("experiment"))
    if not isinstance(objective, dict) or objective.get("loss_mode") != "balanced-regression":
        raise ValueError("训练报告不是正式 balanced-regression 目标")
    if int(training.get("finalEpoch", 0)) <= 0 or not str(training.get("finalModelSha256", "")):
        raise ValueError("训练报告缺少最终轮模型信息")
    for name, metrics in (("dev", dev), ("test", test)):
        values = [metrics.get("macroF1"), metrics.get("macroSpearman"), metrics.get("intensityMae")]
        if int(metrics.get("sampleCount", 0)) <= 0 or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values
        ):
            raise ValueError(f"{name} 指标不完整或包含非有限值")
    if audit.get("passed") is not True:
        raise ValueError("实验验收报告未通过")
    recommendation = audit.get("recommendation")
    selected = recommendation.get("candidate") if isinstance(recommendation, dict) else None
    if not str(selected or "").strip():
        raise ValueError("实验验收报告缺少推荐候选")

    inputs = {
        "trainingReport": _hash(training_report_path),
        "devMetrics": _hash(dev_metrics_path),
        "testMetrics": _hash(test_metrics_path),
        "experimentAudit": _hash(experiment_audit_path),
    }
    return validate_release_approval({
        "schemaVersion": 2,
        "qualityGatePassed": True,
        "selectionPolicy": "experiment-comparison-and-user-selection",
        "selectedCandidate": selected,
        "qualitySummary": {
            "devMacroF1": float(dev["macroF1"]),
            "testMacroF1": float(test["macroF1"]),
            "testMacroSpearman": float(test["macroSpearman"]),
            "testIntensityMae": float(test["intensityMae"]),
        },
        "inputSha256": inputs,
    })
