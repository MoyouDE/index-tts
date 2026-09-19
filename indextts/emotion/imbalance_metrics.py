"""Descriptive continuous-label metrics; no threshold tuning on test."""
from __future__ import annotations

import numpy as np

from .metrics import emotion_metrics
from .schema import EMOTION_NAMES


def error_summary(pred, labels, mask):
    delta = (pred - labels)[mask]
    return {"count": int(delta.size), "mae": float(np.abs(delta).mean()) if delta.size else None,
            "bias": float(delta.mean()) if delta.size else None}


def detailed_metrics(predictions, neutral_threshold):
    x = predictions
    result = emotion_metrics(x['vectors'], x['labels'], x['labelMasks'], x['intensities'],
                             x['predictedIntensities'], threshold=0.35, neutral_threshold=neutral_threshold)
    pred, labels = x['vectors'], x['labels']
    valid = x['labelMasks'] > 0.5
    nonzero_maes = []
    for i, name in enumerate(EMOTION_NAMES):
        p, y, enabled = pred[:, i], labels[:, i], valid[:, i]
        true, actual = (y >= 0.35) & enabled, (p >= 0.35) & enabled
        tp = int((true & actual).sum())
        metrics = result['perEmotion'].setdefault(name, {'f1': 0., 'spearman': 0., 'mae': None})
        metrics.update(precision=tp / max(1, int(actual.sum())), recall=tp / max(1, int(true.sum())),
                       positiveCount=int(true.sum()), predictedPositiveCount=int(actual.sum()))
        metrics['nonzero'] = error_summary(p, y, enabled & (y > 0))
        if metrics['nonzero']['mae'] is not None:
            nonzero_maes.append(metrics['nonzero']['mae'])
        metrics['strengthBins'] = {
            name: error_summary(p, y, enabled & (y > lo) & (y <= hi))
            for name, lo, hi in [('weak', 0, .3), ('medium', .3, .6), ('strong', .6, 1)]}
        metrics['auxiliary'] = error_summary(p, y, enabled & (y > 0) & (y < labels.max(axis=1)))
        zero = enabled & (y == 0)
        metrics['zeroFalseActivationCount'] = int((zero & (p >= .35)).sum())
        metrics['zeroCount'] = int(zero.sum())
        metrics['zeroFalseActivationRate'] = float((p[zero] >= .35).mean()) if zero.any() else None
    zeros = valid & (labels == 0)
    result['macroNonzeroMae'] = float(np.mean(nonzero_maes)) if nonzero_maes else None
    result['vectorMae'] = error_summary(pred, labels, valid)['mae']
    result['zeroDimensionFalseActivationRate'] = float((pred[zeros] >= .35).mean()) if zeros.any() else None
    result['zeroDimensionCount'] = int(zeros.sum())
    result['zeroDimensionFalseActivationCount'] = int(((pred >= .35) & zeros).sum())
    result['auxiliary'] = error_summary(pred, labels, valid & (labels > 0) & (labels < labels.max(axis=1, keepdims=True)))
    neutral = x['intensities'].reshape(-1) <= .05
    intensity = x['predictedIntensities'].reshape(-1)
    result['neutral'] = {'count': int(neutral.sum()),
                         'falseActivationCount': int((intensity[neutral] >= neutral_threshold).sum()),
                         **{'intensityError': error_summary(intensity, x['intensities'].reshape(-1), neutral)}}
    return result


def compare_dev(baseline, candidate):
    weak_f1 = lambda m: (m['perEmotion']['sad']['f1'] + m['perEmotion']['melancholic']['f1']) / 2
    deltas = {k: candidate[k] - baseline[k] for k in
              ['macroF1', 'macroSpearman', 'macroNonzeroMae', 'zeroDimensionFalseActivationRate']}
    deltas['sadMelancholicF1'] = weak_f1(candidate) - weak_f1(baseline)
    improves = deltas['sadMelancholicF1'] >= .02 or deltas['macroNonzeroMae'] <= -.005
    retains = deltas['macroF1'] >= -.01 and deltas['macroSpearman'] >= -.01 and deltas['zeroDimensionFalseActivationRate'] <= .01
    return {'deltas': deltas, 'improvesTarget': improves, 'retainsOverall': retains,
            'worthContinuing': improves and retains}
