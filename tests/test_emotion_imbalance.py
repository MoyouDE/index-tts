from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from indextts.emotion.imbalance import (ImbalanceConfig, dimension_weights, sampling_weights,
                                      mixed_batches, exposure_report, strength_bin)
from indextts.emotion.imbalance_metrics import detailed_metrics, compare_dev
from indextts.emotion.model import EmotionModelOutput, masked_emotion_loss
from indextts.emotion.training_state import epoch_batch_indices


def row(labels):
    return SimpleNamespace(labels=labels, label_mask=[1.] * 8, intensity=max(labels))


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
                                   {'weighted_fraction': 2}, {'sampling_mode': 'unknown'}])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ImbalanceConfig(**kwargs)
