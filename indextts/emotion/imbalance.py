"""Versioned production loss weights and optional reproducible mixed sampling."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .schema import EMOTION_NAMES
from .training_state import epoch_batch_indices


@dataclass(frozen=True)
class ImbalanceConfig:
    implementation: str = "balanced-regression-v1"
    loss_mode: str = "balanced-regression"
    sampling_mode: str = "uniform"
    regression_beta: float = 0.1
    regression_weight: float = 1.0
    weighted_fraction: float = 0.2
    dimension_formula: str = "clip(sqrt(negative/positive),1,3);mean-normalized"
    sampling_formula: str = "max(clip(sqrt(N/bin_count),1,3));zero=3"

    def __post_init__(self):
        if self.implementation != "balanced-regression-v1" or self.loss_mode != "balanced-regression":
            raise ValueError("Unknown training objective implementation or loss")
        if self.sampling_mode not in {"uniform", "mixed"}:
            raise ValueError("Unknown sampling mode")
        if not all(math.isfinite(v) for v in (self.regression_beta, self.regression_weight)) or self.regression_beta <= 0 or self.regression_weight < 0:
            raise ValueError("Invalid regression parameters")
        if not 0 <= self.weighted_fraction <= 1:
            raise ValueError("Invalid sampling fraction")
        if self.dimension_formula != type(self).dimension_formula or self.sampling_formula != type(self).sampling_formula:
            raise ValueError("Weight formula must match the versioned implementation")

    def as_dict(self):
        return asdict(self)


def strength_bin(value):
    return 0 if value <= 0.3 else (1 if value <= 0.6 else 2)


def dimension_weights(examples):
    observed = np.zeros(8)
    positives = np.zeros(8)
    for row in examples:
        if row.intensity <= 1e-6:
            continue
        mask = np.asarray(row.label_mask) > 0.5
        observed += mask
        positives += (np.asarray(row.labels) > 0) & mask
    raw = np.ones(8)
    valid = positives > 0
    raw[valid] = np.clip(np.sqrt((observed[valid] - positives[valid]) / positives[valid]), 1, 3)
    weights = raw / raw.mean()
    return torch.tensor(weights, dtype=torch.float32), {
        "observed": dict(zip(EMOTION_NAMES, observed.astype(int).tolist())),
        "positive": dict(zip(EMOTION_NAMES, positives.astype(int).tolist())),
        "missingPositiveDimensions": [d for i, d in enumerate(EMOTION_NAMES) if not valid[i]],
        "rawWeights": raw.tolist(), "normalizedWeights": weights.tolist(),
    }


def sampling_weights(examples):
    counts = np.zeros((8, 3), dtype=np.int64)
    for row in examples:
        for d, (v, mask) in enumerate(zip(row.labels, row.label_mask)):
            if v > 0 and mask > 0.5:
                counts[d, strength_bin(v)] += 1
    weights = []
    for row in examples:
        values = [min(3.0, max(1.0, (len(examples) / counts[d, strength_bin(v)]) ** 0.5))
                  for d, (v, mask) in enumerate(zip(row.labels, row.label_mask)) if v > 0 and mask > 0.5]
        weights.append(max(values) if values else 3.0)
    return torch.tensor(weights, dtype=torch.float64), counts.tolist()


def mixed_batches(weights, batch_size, seed, epoch, fraction=0.2):
    n = len(weights)
    if n == 0 or batch_size <= 0 or not 0 <= fraction <= 1:
        raise ValueError("Invalid batch parameters")
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sampling weights must be positive and finite")
    if fraction == 0:
        return epoch_batch_indices(n, batch_size, seed, epoch)
    generator = torch.Generator().manual_seed(seed + epoch - 1)
    k = round(n * fraction)
    uniform = torch.randperm(n, generator=generator)[:n-k]
    weighted = torch.multinomial(weights, k, replacement=True, generator=generator) if k else torch.empty(0, dtype=torch.long)
    indices = torch.cat((uniform, weighted))
    indices = indices[torch.randperm(n, generator=generator)].tolist()
    return [indices[i:i+batch_size] for i in range(0, n, batch_size)]


def exposure_report(examples, batches):
    indices = [i for batch in batches for i in batch]
    repeats = Counter(indices)
    bins = np.zeros((8, 3), dtype=np.int64)
    zeros = 0
    for i in indices:
        row = examples[i]
        zeros += int(max(row.labels) == 0)
        for d, (v, mask) in enumerate(zip(row.labels, row.label_mask)):
            if v > 0 and mask > 0.5:
                bins[d, strength_bin(v)] += 1
    return {"sampleCount": len(indices), "uniqueSamples": len(repeats),
            "duplicateDraws": len(indices)-len(repeats),
            "maxMultiplicity": max(repeats.values(), default=0),
            "multiplicityHistogram": dict(sorted(Counter(repeats.values()).items())),
            "zeroExamples": zeros, "dimensionExposure": dict(zip(EMOTION_NAMES, bins.sum(axis=1).tolist())),
            "strengthBins": dict(zip(EMOTION_NAMES, bins.tolist()))}
