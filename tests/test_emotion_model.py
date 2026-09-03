from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from indextts.emotion.cli import _release_data_check
from indextts.emotion.dataset import format_current_text
from indextts.emotion.metrics import calibrate_neutral_threshold, emotion_metrics
from indextts.emotion.model import MacBertEmotionModel, masked_emotion_loss
from indextts.emotion.schema import EMOTION_NAMES, parse_example, validate_work_splits


def _record(**overrides):
    record = {
        "schemaVersion": 1,
        "id": "work-1-1",
        "workId": "work-1",
        "previousText": "夜色渐深。",
        "text": "他终于回来了。",
        "sentenceType": "narration",
        "emotions": {name: 0.0 for name in EMOTION_NAMES},
        "labelMask": {name: 1.0 for name in EMOTION_NAMES},
        "intensity": 0.0,
        "licenseId": "PROPRIETARY-AUTHORIZED",
        "source": "readest-human-annotation-v1",
    }
    record.update(overrides)
    return record


def test_schema_requires_commercial_license_and_keeps_neutral_distinct_from_calm():
    example = parse_example(_record())
    assert example.intensity == 0.0
    assert example.labels == (0.0,) * 8
    assert example.label_mask == (1.0,) * 8
    assert example.license_status == "approved"

    with pytest.raises(ValueError, match="商用白名单"):
        parse_example(_record(licenseId="CC-BY-NC-4.0"))


def test_release_data_check_rejects_test_only_novel_examples():
    examples = [
        parse_example(
            _record(
                id=f"work-{index}-1",
                workId=f"work-{index}",
                licenseStatus="test-only",
            )
        )
        for index in range(3)
    ]
    report = _release_data_check(
        examples[0:1] * 4000 + examples[1:2] * 4000 + examples[2:3] * 4000
    )
    assert report["releaseEligible"] is False
    assert report["authorizedNovelExamples"] == 0
    assert report["testOnlyNovelExamples"] == 12_000


def test_release_data_check_rejects_qwen_selected_or_low_confidence_labels():
    approved = [
        parse_example(
            _record(
                id=f"approved-{index}",
                workId=f"approved-work-{index}",
                licenseStatus="approved",
                adjudicationChosenSource="agent",
                adjudicationConfidence="high",
            )
        )
        for index in range(3)
    ]
    eligible = _release_data_check(
        approved[0:1] * 4000 + approved[1:2] * 4000 + approved[2:3] * 4000
    )
    assert eligible["releaseEligible"] is True

    qwen = parse_example(
        _record(
            id="qwen-selected",
            workId="approved-work-0",
            licenseStatus="approved",
            adjudicationChosenSource="qwen",
            adjudicationConfidence="low",
        )
    )
    blocked = _release_data_check([qwen, *approved[0:1] * 3999, *approved[1:2] * 4000, *approved[2:3] * 4000])
    assert blocked["releaseEligible"] is False
    assert blocked["qwenSelectedNovelExamples"] == 1
    assert blocked["lowConfidenceNovelExamples"] == 1


def test_schema_rejects_work_leakage_across_splits():
    example = parse_example(_record())
    with pytest.raises(ValueError, match="存在泄漏"):
        validate_work_splits({"train": [example], "dev": [example]})


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        del attention_mask, token_type_ids, return_dict
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return SimpleNamespace(last_hidden_state=self.projection(hidden))


def test_model_outputs_eight_strengths_and_independent_intensity_gate():
    model = MacBertEmotionModel(_FakeEncoder(), 4, dropout=0.0)
    inputs = torch.tensor([[1, 2, 3], [3, 2, 1]])
    output = model(inputs, torch.ones_like(inputs), torch.zeros_like(inputs))
    assert output.emotion_logits.shape == (2, 8)
    assert output.intensity.shape == (2, 1)
    assert output.emotion_vector.shape == (2, 8)
    assert torch.all(output.emotion_vector >= 0)
    assert torch.all(output.emotion_vector <= 1)

    labels = torch.zeros((2, 8))
    mask = torch.ones((2, 8))
    mask[:, 5] = 0
    mask[:, 7] = 0
    loss, parts = masked_emotion_loss(output, labels, mask, torch.zeros((2, 1)))
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "emotionLoss", "intensityLoss"}


def test_loss_trains_conditional_composition_before_applying_intensity_gate():
    class FixedOutput:
        emotion_logits = torch.tensor([[2.0, 0, 0, 0, 0, 0, 0, 0]])
        intensity_logits = torch.zeros((1, 1))

    labels = torch.tensor([[0.5, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    mask = torch.ones((1, 8), dtype=torch.float32)
    intensity = torch.tensor([[0.5]], dtype=torch.float32)
    loss, parts = masked_emotion_loss(FixedOutput(), labels, mask, intensity)
    expected_targets = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        FixedOutput.emotion_logits, expected_targets
    )
    assert torch.isfinite(loss)
    assert parts["emotionLoss"] == pytest.approx(float(expected))


def test_neutral_rows_train_only_the_independent_intensity_gate():
    class FixedOutput:
        emotion_logits = torch.full((1, 8), 5.0)
        intensity_logits = torch.zeros((1, 1))

    labels = torch.zeros((1, 8), dtype=torch.float32)
    mask = torch.ones((1, 8), dtype=torch.float32)
    intensity = torch.zeros((1, 1), dtype=torch.float32)
    _, parts = masked_emotion_loss(FixedOutput(), labels, mask, intensity)
    assert parts["emotionLoss"] == 0.0


def test_neutral_loss_weight_penalizes_false_active_gate_predictions():
    class FixedOutput:
        emotion_logits = torch.zeros((2, 8))
        intensity_logits = torch.full((2, 1), 2.0)

    labels = torch.zeros((2, 8), dtype=torch.float32)
    mask = torch.ones((2, 8), dtype=torch.float32)
    intensity = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    _, unweighted = masked_emotion_loss(
        FixedOutput(), labels, mask, intensity, neutral_loss_weight=1.0
    )
    _, weighted = masked_emotion_loss(
        FixedOutput(), labels, mask, intensity, neutral_loss_weight=3.0
    )
    assert weighted["intensityLoss"] > unweighted["intensityLoss"]


def test_positive_weights_balance_only_active_unmasked_examples():
    from indextts.emotion.schema import EmotionExample
    from indextts.emotion.train import emotion_positive_weights

    def example(example_id, labels, intensity, mask=None):
        return EmotionExample(
            example_id=example_id,
            work_id="work",
            previous_text="previous",
            text="text",
            sentence_type="narration",
            labels=tuple(labels),
            label_mask=tuple(mask or [1.0] * 8),
            intensity=intensity,
            license_id="PROPRIETARY-AUTHORIZED",
            source="test",
        )

    rows = [
        example("positive", [1.0, 0, 0, 0, 0, 0, 0, 0], 1.0),
        example("negative-a", [0.0] * 8, 1.0),
        example("negative-b", [0.0] * 8, 1.0),
        example("neutral-ignored", [0.0] * 8, 0.0),
    ]
    weights = emotion_positive_weights(rows)
    assert float(weights[0]) == pytest.approx(2.0)
    assert torch.all(weights[1:] == 1.0)


def test_metrics_ignore_masked_dimensions_and_measure_neutral_activation():
    labels = np.zeros((2, 8), dtype=np.float32)
    labels[1, 0] = 1.0
    predictions = labels.copy()
    predictions[0] = 0.2
    mask = np.ones((2, 8), dtype=np.float32)
    mask[:, 5] = 0
    mask[:, 7] = 0
    metrics = emotion_metrics(
        predictions,
        labels,
        mask,
        np.array([[0.0], [1.0]]),
        np.array([[0.2], [0.9]]),
    )
    assert metrics["sampleCount"] == 2
    assert metrics["neutralFalseActivationRate"] == 1.0
    assert "melancholic" not in metrics["perEmotion"]
    assert "calm" not in metrics["perEmotion"]


def test_neutral_threshold_calibration_uses_balanced_accuracy():
    report = calibrate_neutral_threshold(
        np.asarray([0.0, 0.0, 0.67, 0.67]),
        np.asarray([0.1, 0.2, 0.4, 0.8]),
        minimum=0.1,
        maximum=0.5,
        step=0.1,
    )
    assert report["activeExamples"] == 2
    assert report["baseExamples"] == 2
    assert report["recommended"]["threshold"] == pytest.approx(0.4)
    assert report["recommended"]["balancedAccuracy"] == 1.0


def test_sentence_type_prefix_is_stable():
    assert format_current_text("你好。", "dialogue") == "[对白]你好。"
    assert format_current_text("天亮了。", "narration") == "[旁白]天亮了。"
