from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from indextts.emotion.imbalance import (
    ImbalanceConfig, NARRATION_SAMPLING_FORMULA, dimension_weights,
    sampling_weights, mixed_batches, narration_mix_batches, exposure_report,
    strength_bin,
)
from indextts.emotion.imbalance_metrics import detailed_metrics, compare_dev
from indextts.emotion.model import EmotionModelOutput, masked_emotion_loss
from indextts.emotion.training_state import epoch_batch_indices


def row(labels, sentence_type='dialogue'):
    return SimpleNamespace(labels=labels, label_mask=[1.] * 8,
                           intensity=max(labels), sentence_type=sentence_type)


@pytest.mark.parametrize('labels', [[0.] * 8, [1.] * 8, [0., .2, .7, 0., 1., .01, 0., .3]])
def test_balanced_loss_finite_and_backprop(labels):
    y = torch.tensor([labels])
    out = EmotionModelOutput(torch.zeros((1, 8), requires_grad=True), torch.zeros((1, 1), requires_grad=True))
    loss, parts = masked_emotion_loss(out, y, torch.ones_like(y), y.max(1, keepdim=True).values,
                                     dimension_weights=torch.ones(8), regression_weight=1., intensity_weight=.7,
                                     neutral_loss_weight=3.)
    assert torch.isfinite(loss) and 'regressionLoss' in parts
    loss.backward()
    assert torch.isfinite(out.emotion_logits.grad).all()
    assert torch.isfinite(out.intensity_logits.grad).all()
    if max(labels) == 0:
        assert parts['emotionLoss'] == 0
        assert out.emotion_logits.grad.abs().sum() > 0  # regression includes zero rows


def test_default_loss_preserved_exactly():
    y = torch.tensor([[.2, .8, 0., 0., .4, 0., 0., 0.], [0.] * 8])
    intensity = y.max(1, keepdim=True).values
    mask = torch.ones_like(y)
    out = EmotionModelOutput(torch.arange(16).reshape(2, 8).float() / 8, torch.tensor([[.2], [-.3]]))
    weights = torch.arange(1, 9).float()
    loss, parts = masked_emotion_loss(out, y, mask, intensity, positive_weights=weights,
                                     intensity_weight=.7, neutral_loss_weight=3)
    conditional = torch.where(intensity > 1e-6, y / intensity.clamp_min(1e-6), 0.).clamp(0., 1.)
    active = mask * (intensity > 1e-6)
    bce = F.binary_cross_entropy_with_logits(out.emotion_logits, conditional, pos_weight=weights, reduction='none')
    gate = F.binary_cross_entropy_with_logits(out.intensity_logits, intensity, reduction='none')
    expected = (bce * active).sum() / active.sum() + .7 * (gate * torch.tensor([[1.], [3.]])).sum() / 4
    assert torch.equal(loss, expected)
    assert 'regressionLoss' not in parts


def test_positive_aware_regression_weights_positive_and_auxiliary_elements():
    labels = torch.tensor([[.8, .2, 0., 0., 0., 0., 0., 0.]])
    mask = torch.ones_like(labels)
    intensity = labels.max(1, keepdim=True).values
    output = EmotionModelOutput(torch.zeros((1, 8)), torch.zeros((1, 1)))
    _, uniform = masked_emotion_loss(
        output,
        labels,
        mask,
        intensity,
        regression_weight=1.0,
    )
    _, weighted = masked_emotion_loss(
        output,
        labels,
        mask,
        intensity,
        regression_weight=1.0,
        regression_positive_weight=2.0,
        regression_auxiliary_weight=1.5,
    )
    raw = F.smooth_l1_loss(output.emotion_vector, labels, beta=.1, reduction='none')
    expected_weights = torch.tensor([[2., 3., 1., 1., 1., 1., 1., 1.]])
    expected = (raw * expected_weights).sum() / expected_weights.sum()
    assert weighted['regressionLoss'] == pytest.approx(expected.item())
    assert weighted['regressionLoss'] != uniform['regressionLoss']


def test_positive_aware_config_is_versioned_and_v1_stays_uniform():
    config = ImbalanceConfig(
        implementation='positive-aware-regression-v2',
        loss_mode='balanced-positive-aware-regression',
        regression_positive_weight=2.0,
        regression_auxiliary_weight=1.5,
        regression_formula='positive-multiplier;auxiliary-positive-multiplier;weighted-mean',
    )
    assert config.as_dict()['regression_positive_weight'] == 2.0
    with pytest.raises(ValueError, match='v1 requires'):
        ImbalanceConfig(regression_positive_weight=2.0)


@pytest.mark.parametrize('target', [.1, .35, .8])
def test_whole_bce_weight_preserves_soft_optimum(target):
    logits = torch.logit(torch.tensor([target], dtype=torch.float64)).requires_grad_()
    loss = 2.7 * F.binary_cross_entropy_with_logits(logits, torch.tensor([target], dtype=torch.float64))
    loss.backward()
    assert abs(logits.grad.item()) < 1e-12


def test_train_weights_and_sampler():
    examples = [row([.2, .7, 0., 0., 0., 0., 0., 0.])] + [row([.4] + [0.] * 7)] * 18 + [row([0.] * 8)]
    weights, report = dimension_weights(examples)
    assert weights.mean().item() == pytest.approx(1.)
    assert report['rawWeights'][0] == 1 and report['rawWeights'][1] == 3
    assert len(report['missingPositiveDimensions']) == 6
    sample, bins = sampling_weights(examples)
    assert sample.min() >= 1 and sample.max() <= 3
    assert sample[0] == 3 and sample[-1] == 3
    assert sample[1] == pytest.approx((20 / 18) ** .5)
    assert bins[0] == [1, 18, 0]
    assert [strength_bin(v) for v in [.01, .3, .31, .6, .61, 1]] == [0, 0, 1, 1, 2, 2]
    batches = mixed_batches(sample, 6, 123, 1)
    assert batches == mixed_batches(sample, 6, 123, 1)
    assert batches != mixed_batches(sample, 6, 123, 2)
    assert mixed_batches(sample, 6, 123, 1, 0) == epoch_batch_indices(20, 6, 123, 1)
    report = exposure_report(examples, batches)
    assert report['sampleCount'] == 20
    assert report['uniqueSamples'] + report['duplicateDraws'] == 20
    assert sum(len(b) for b in batches) == 20


def test_narration_mix_is_deterministic_and_preserves_epoch_size():
    examples = [row([.5] + [0.] * 7) for _ in range(90)]
    examples += [row([.4] + [0.] * 7, 'narration') for _ in range(10)]
    batches = narration_mix_batches(examples, 6, 123, 1, .2)
    assert batches == narration_mix_batches(examples, 6, 123, 1, .2)
    assert batches != narration_mix_batches(examples, 6, 123, 2, .2)
    assert narration_mix_batches(examples, 6, 123, 1, 0) == epoch_batch_indices(100, 6, 123, 1)
    assert sum(len(batch) for batch in batches) == 100
    report = exposure_report(examples, batches)
    assert report['sampleCount'] == 100
    assert report['sentenceTypeExposure']['narration'] >= 20
    assert report['uniqueSamples'] + report['duplicateDraws'] == 100
    assert report['maxMultiplicity'] <= 21


def test_narration_mix_config_is_versioned_and_default_fingerprint_compatible():
    default = ImbalanceConfig().as_dict()
    assert 'narration_fraction' not in default
    config = ImbalanceConfig(
        implementation='positive-aware-regression-v2',
        loss_mode='balanced-positive-aware-regression',
        regression_positive_weight=2.0,
        regression_auxiliary_weight=1.5,
        regression_formula='positive-multiplier;auxiliary-positive-multiplier;weighted-mean',
        sampling_mode='narration-mix',
        narration_fraction=.05,
        sampling_formula=NARRATION_SAMPLING_FORMULA,
    )
    assert config.as_dict()['narration_fraction'] == .05
    assert config.as_dict()['sampling_formula'] == NARRATION_SAMPLING_FORMULA
    assert config.as_dict() != default
    with pytest.raises(ValueError, match='Narration sampling'):
        ImbalanceConfig(sampling_mode='narration-mix', narration_fraction=.05)
    with pytest.raises(ValueError, match='Non-narration sampling'):
        ImbalanceConfig(narration_fraction=.05)
    with pytest.raises(ValueError, match='requires narration examples'):
        narration_mix_batches([row([0.] * 8)], 1, 123, 1, .5)


def test_metrics_zero_auxiliary_bins():
    y = np.array([[.8, .2, 0, 0, 0, 0, 0, 0], [0.] * 8])
    p = np.array([[.7, .3, .4, 0, 0, 0, 0, 0], [0.] * 8])
    m = detailed_metrics(dict(vectors=p, labels=y, labelMasks=np.ones_like(y),
                              intensities=np.array([.8, 0]), predictedIntensities=np.array([.7, .4])), .325)
    assert m['auxiliary']['count'] == 1 and m['auxiliary']['mae'] == pytest.approx(.1)
    assert m['zeroDimensionCount'] == 14 and m['zeroDimensionFalseActivationCount'] == 1
    assert m['neutral']['count'] == 1 and m['neutral']['falseActivationCount'] == 1
    assert m['macroNonzeroMae'] == pytest.approx(.1)
    assert not compare_dev(m, m)['worthContinuing']


@pytest.mark.parametrize('kwargs', [{'regression_beta': 0}, {'regression_weight': -1},
                                   {'regression_positive_weight': 0},
                                   {'regression_auxiliary_weight': 0},
                                   {'weighted_fraction': 2}, {'sampling_mode': 'unknown'}])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ImbalanceConfig(**kwargs)
