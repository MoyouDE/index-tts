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
        "macroF1": float(np.mean(f1_values)) if f1_values else 0.0,
        "macroSpearman": float(np.mean(correlations)) if correlations else 0.0,
        "intensityMae": float(np.abs(intensities - predicted_intensities).mean()),
        "neutralFalseActivationRate": false_activation,
        "perEmotion": per_emotion,
        "sampleCount": int(predictions.shape[0]),
    }


def metrics_are_finite(metrics: dict[str, object]) -> bool:
    values = [metrics.get("macroF1"), metrics.get("macroSpearman"), metrics.get("intensityMae")]
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)
