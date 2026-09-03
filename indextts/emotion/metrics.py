"""Dependency-light metrics used by training, release gates and reports."""

from __future__ import annotations

import math

import numpy as np

from .schema import EMOTION_NAMES


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            positions = np.flatnonzero(inverse == group)
            ranks[positions] = ranks[positions].mean()
    return ranks


def _binary_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    tp = int(np.logical_and(target, prediction).sum())
    fp = int(np.logical_and(np.logical_not(target), prediction).sum())
    fn = int(np.logical_and(target, np.logical_not(prediction)).sum())
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def emotion_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    intensities: np.ndarray,
    predicted_intensities: np.ndarray,
    *,
    threshold: float = 0.35,
    neutral_threshold: float = 0.15,
) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    label_mask = np.asarray(label_mask, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    predicted_intensities = np.asarray(predicted_intensities, dtype=np.float64).reshape(-1)
    if predictions.shape != labels.shape or labels.shape != label_mask.shape:
        raise ValueError("predictions/labels/label_mask 形状必须一致")
    if predictions.ndim != 2 or predictions.shape[1] != len(EMOTION_NAMES):
        raise ValueError("情感指标输入必须是 [N, 8]")

    per_emotion: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    correlations: list[float] = []
    for index, name in enumerate(EMOTION_NAMES):
        enabled = label_mask[:, index] > 0.5
        if not enabled.any():
            continue
        expected = labels[enabled, index]
        actual = predictions[enabled, index]
        f1 = _binary_f1(expected >= threshold, actual >= threshold)
        spearman = _pearson(_rank(expected), _rank(actual))
        mae = float(np.abs(expected - actual).mean())
        per_emotion[name] = {"f1": f1, "spearman": spearman, "mae": mae}
        f1_values.append(f1)
        correlations.append(spearman)

    neutral = intensities <= 0.05
    false_activation = (
        float((predicted_intensities[neutral] >= neutral_threshold).mean())
        if neutral.any()
        else 0.0
    )
    return {
        "emotionThreshold": float(threshold),
        "neutralThreshold": float(neutral_threshold),
        "macroF1": float(np.mean(f1_values)) if f1_values else 0.0,
        "macroSpearman": float(np.mean(correlations)) if correlations else 0.0,
        "intensityMae": float(np.abs(intensities - predicted_intensities).mean()),
        "neutralFalseActivationRate": false_activation,
        "perEmotion": per_emotion,
        "sampleCount": int(predictions.shape[0]),
    }


def calibrate_neutral_threshold(
    intensities: np.ndarray,
    predicted_intensities: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 0.8,
    step: float = 0.005,
    minimum_active_recall: float = 0.8,
) -> dict[str, object]:
    """Minimize base false activation while retaining required active recall."""

    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("校准阈值范围必须位于 [0, 1] 且 minimum <= maximum")
    if step <= 0.0:
        raise ValueError("校准步长必须为正数")
    if not 0.0 <= minimum_active_recall <= 1.0:
        raise ValueError("minimum_active_recall 必须位于 [0, 1]")
    expected = np.asarray(intensities, dtype=np.float64).reshape(-1) > 0.05
    predicted = np.asarray(predicted_intensities, dtype=np.float64).reshape(-1)
    if expected.shape != predicted.shape or expected.size == 0:
        raise ValueError("校准输入必须是长度一致的非空数组")
    active_count = int(expected.sum())
    base_count = int((~expected).sum())
    if not active_count or not base_count:
        raise ValueError("校准集必须同时包含 active 与 base 样本")

    thresholds = np.arange(minimum, maximum + step * 0.5, step, dtype=np.float64)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        actual = predicted >= threshold
        true_positive = int(np.logical_and(expected, actual).sum())
        false_positive = int(np.logical_and(~expected, actual).sum())
        false_negative = active_count - true_positive
        recall = true_positive / active_count
        false_activation = false_positive / base_count
        specificity = 1.0 - false_activation
        precision_denominator = true_positive + false_positive
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        f1_denominator = 2 * true_positive + false_positive + false_negative
        active_f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
        rows.append(
            {
                "threshold": float(round(threshold, 10)),
                "balancedAccuracy": (recall + specificity) / 2.0,
                "activeRecall": recall,
                "activePrecision": precision,
                "activeF1": active_f1,
                "neutralFalseActivationRate": false_activation,
            }
        )
    eligible = [row for row in rows if row["activeRecall"] >= minimum_active_recall]
    if not eligible:
        maximum_recall = max(row["activeRecall"] for row in rows)
        raise ValueError(
            "校准范围内无法满足主动情感召回率要求: "
            f"required={minimum_active_recall:g}, maximum={maximum_recall:g}"
        )
    best = min(
        eligible,
        key=lambda row: (
            row["neutralFalseActivationRate"],
            -row["balancedAccuracy"],
            -row["threshold"],
        ),
    )
    return {
        "selectionPolicy": "minimum-neutral-far-subject-to-active-recall",
        "minimumActiveRecall": minimum_active_recall,
        "activeExamples": active_count,
        "baseExamples": base_count,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "recommended": best,
        "curve": rows,
    }


def metrics_are_finite(metrics: dict[str, object]) -> bool:
    values = [metrics.get("macroF1"), metrics.get("macroSpearman"), metrics.get("intensityMae")]
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
