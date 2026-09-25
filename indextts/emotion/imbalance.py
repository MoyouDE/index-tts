"""Versioned production loss weights and optional reproducible mixed sampling."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .schema import EMOTION_NAMES
from .training_state import epoch_batch_indices

NARRATION_SAMPLING_FORMULA = "uniform-without-replacement+narration-with-replacement;total=N"


@dataclass(frozen=True)
class ImbalanceConfig:
    implementation: str = "balanced-regression-v1"
    loss_mode: str = "balanced-regression"
    sampling_mode: str = "uniform"
    regression_beta: float = 0.1
    regression_weight: float = 1.0
    regression_positive_weight: float = 1.0
    regression_auxiliary_weight: float = 1.0
    weighted_fraction: float = 0.2
    narration_fraction: float = 0.0
    dimension_formula: str = "clip(sqrt(negative/positive),1,3);mean-normalized"
    sampling_formula: str = "max(clip(sqrt(N/bin_count),1,3));zero=3"
    regression_formula: str = "uniform-valid-dimensions"

    def __post_init__(self):
        implementations = {
            "balanced-regression-v1": (
                "balanced-regression",
                "uniform-valid-dimensions",
            ),
            "positive-aware-regression-v2": (
                "balanced-positive-aware-regression",
                "positive-multiplier;auxiliary-positive-multiplier;weighted-mean",
            ),
        }
        if self.implementation not in implementations:
            raise ValueError("Unknown training objective implementation or loss")
        expected_loss, expected_formula = implementations[self.implementation]
        if self.loss_mode != expected_loss or self.regression_formula != expected_formula:
            raise ValueError("Training objective fields do not match implementation")
        if self.sampling_mode not in {"uniform", "mixed", "narration-mix"}:
            raise ValueError("Unknown sampling mode")
        regression_values = (
            self.regression_beta,
            self.regression_weight,
            self.regression_positive_weight,
            self.regression_auxiliary_weight,
        )
        if (
            not all(math.isfinite(v) for v in regression_values)
            or self.regression_beta <= 0
            or self.regression_weight < 0
            or self.regression_positive_weight <= 0
            or self.regression_auxiliary_weight <= 0
        ):
            raise ValueError("Invalid regression parameters")
        if self.implementation == "balanced-regression-v1" and (
            self.regression_positive_weight != 1.0
            or self.regression_auxiliary_weight != 1.0
        ):
            raise ValueError("v1 requires uniform regression weights")
        if not 0 <= self.weighted_fraction <= 1:
            raise ValueError("Invalid sampling fraction")
        if not math.isfinite(self.narration_fraction) or not 0 <= self.narration_fraction < 1:
            raise ValueError("Invalid narration sampling fraction")
        if self.sampling_mode == "narration-mix":
            if self.narration_fraction <= 0 or self.sampling_formula != NARRATION_SAMPLING_FORMULA:
                raise ValueError("Narration sampling configuration does not match implementation")
        elif self.narration_fraction != 0 or self.sampling_formula != type(self).sampling_formula:
            raise ValueError("Non-narration sampling requires its versioned defaults")
        if self.dimension_formula != type(self).dimension_formula:
            raise ValueError("Weight formula must match the versioned implementation")

    def as_dict(self):
        value = asdict(self)
        if self.sampling_mode != "narration-mix":
            value.pop("narration_fraction")
        return value


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


def narration_mix_batches(examples, batch_size, seed, epoch, fraction):
    n = len(examples)
    if n == 0 or batch_size <= 0 or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Invalid narration batch parameters")
    if fraction == 0:
        return epoch_batch_indices(n, batch_size, seed, epoch)
    narration = torch.tensor(
        [index for index, row in enumerate(examples) if row.sentence_type == "narration"],
        dtype=torch.long,
    )
    if not len(narration):
        raise ValueError("Narration sampling requires narration examples")
    if round(n * fraction) == 0:
        return epoch_batch_indices(n, batch_size, seed, epoch)
    generator = torch.Generator().manual_seed(seed + epoch - 1)
    k = round(n * fraction)
    uniform = torch.randperm(n, generator=generator)[:n-k]
    extra = narration[torch.randint(len(narration), (k,), generator=generator)]
    indices = torch.cat((uniform, extra))
    indices = indices[torch.randperm(n, generator=generator)].tolist()
    return [indices[index:index + batch_size] for index in range(0, n, batch_size)]


def exposure_report(examples, batches):
    indices = [i for batch in batches for i in batch]
    repeats = Counter(indices)
    sentence_types = Counter(examples[i].sentence_type for i in indices)
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
            "sentenceTypeExposure": dict(sorted(sentence_types.items())),
            "maxMultiplicity": max(repeats.values(), default=0),
            "multiplicityHistogram": dict(sorted(Counter(repeats.values()).items())),
            "zeroExamples": zeros, "dimensionExposure": dict(zip(EMOTION_NAMES, bins.sum(axis=1).tolist())),
            "strengthBins": dict(zip(EMOTION_NAMES, bins.tolist()))}
