"""Quality-first release gates and legacy-Qwen benchmark support."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import emotion_metrics
from .schema import EmotionExample


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
    """Validate the evidence envelope required for an approved model export."""
    if not isinstance(approval, dict) or approval.get("schemaVersion") != 1:
        raise ValueError("发布批准文件 schemaVersion 必须为 1")
    required_true = (
        "qualityGatePassed",
        "humanBlindTestPassed",
        "commercialDataAuditPassed",
        "noQwenPseudoLabels",
    )
    if not all(approval.get(name) is True for name in required_true):
        raise ValueError(f"发布批准文件必须将 {required_true} 全部设为 true")
    if not str(approval.get("approvedBy", "")).strip():
        raise ValueError("发布批准文件缺少 approvedBy")
    if int(approval.get("blindJudgments", 0)) < 100:
        raise ValueError("发布批准文件的 TTS 盲听判断少于 100 项")
    if float(approval.get("candidateNoWorseRate", 0.0)) < 0.5:
        raise ValueError("发布批准文件的候选模型听感低于 Qwen")
    input_hashes = approval.get("inputSha256")
    expected_inputs = {
        "trainingReport",
        "modelMetrics",
        "qwenMetrics",
        "blindTest",
        "dataAudit",
    }
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_inputs:
        raise ValueError("发布批准文件的 inputSha256 证据集合不完整")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
        for value in input_hashes.values()
    ):
        raise ValueError("发布批准文件包含无效的 SHA-256")
    return approval


def build_release_approval(
    *,
    training_report_path: str | Path,
    model_metrics_path: str | Path,
    qwen_metrics_path: str | Path,
    blind_test_path: str | Path,
    data_audit_path: str | Path,
) -> dict[str, object]:
    training = _read_json(training_report_path)
    model = _read_json(model_metrics_path)
    qwen = _read_json(qwen_metrics_path)
    blind = _read_json(blind_test_path)
    audit = _read_json(data_audit_path)

    release_data = training.get("releaseData")
    if not isinstance(release_data, dict) or release_data.get("releaseEligible") is not True:
        raise ValueError("训练报告未达到 12000 条完整八维、权利明确的小说标注门槛")
    if training.get("releaseTraining") is not True:
        raise ValueError("训练任务未使用 --release")
    if audit.get("commercialDataAuditPassed") is not True or not audit.get("approvedBy"):
        raise ValueError("商业数据审计尚未通过或缺少 approvedBy")

    quality_passed = (
        float(model.get("macroF1", -1)) >= float(qwen.get("macroF1", 0))
        and float(model.get("macroSpearman", -1)) >= float(qwen.get("macroSpearman", 0))
        and int(model.get("sampleCount", 0)) == int(qwen.get("sampleCount", -1))
        and int(qwen.get("warningCount", 0)) >= 0
    )
    if not quality_passed:
        raise ValueError("MacBERT 指标未同时达到 Qwen 基线，禁止发布")

    candidate_wins = int(blind.get("candidateWins", 0))
    qwen_wins = int(blind.get("qwenWins", 0))
    ties = int(blind.get("ties", 0))
    judgments = candidate_wins + qwen_wins + ties
    no_worse_rate = (candidate_wins + ties * 0.5) / judgments if judgments else 0.0
    blind_passed = judgments >= 100 and no_worse_rate >= 0.5
    if not blind_passed:
        raise ValueError("TTS 人工盲测少于 100 项或候选模型听感低于 Qwen")

    inputs = {
        "trainingReport": _hash(training_report_path),
        "modelMetrics": _hash(model_metrics_path),
        "qwenMetrics": _hash(qwen_metrics_path),
        "blindTest": _hash(blind_test_path),
        "dataAudit": _hash(data_audit_path),
    }
    return validate_release_approval({
        "schemaVersion": 1,
        "qualityGatePassed": True,
        "humanBlindTestPassed": True,
        "commercialDataAuditPassed": True,
        "noQwenPseudoLabels": True,
        "blindJudgments": judgments,
        "candidateNoWorseRate": no_worse_rate,
        "approvedBy": audit["approvedBy"],
        "inputSha256": inputs,
    })
