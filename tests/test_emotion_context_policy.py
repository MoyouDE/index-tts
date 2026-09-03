from __future__ import annotations

import json
from pathlib import Path

import pytest

from indextts.emotion.context_policy import select_complete_context
from indextts.emotion.continuous_v3_context import _expand_record
from indextts.emotion.training_preflight import _complete_context_summary


def _count(previous_text: str) -> int:
    return 4 + len(previous_text)


def test_selects_nearest_three_complete_sentences_in_chronological_order():
    selected = select_complete_context(["zero", " one ", "two", "three"], _count, max_length=20)

    assert selected.previous_sentences == ("one", "two", "three")
    assert selected.previous_text == "one\ntwo\nthree"
    assert selected.input_token_count == 17
    assert selected.stop_reason == "context-limit"


def test_stops_when_adding_the_third_sentence_would_overflow():
    selected = select_complete_context(["123456", "abcd", "efgh"], _count, max_length=14)

    assert selected.previous_sentences == ("abcd", "efgh")
    assert selected.previous_text == "abcd\nefgh"
    assert selected.context_limited is True
    assert selected.stop_reason == "context-over-max"


def test_does_not_skip_an_overlong_nearest_sentence_for_an_older_one():
    selected = select_complete_context(["old", "nearest-is-too-long"], _count, max_length=12)

    assert selected.previous_sentences == ()
    assert selected.previous_text == ""
    assert selected.context_limited is True


def test_context_whitespace_is_normalized_without_partial_sentences():
    selected = select_complete_context(
        ["  first\t sentence  ", "   ", "second   sentence"],
        lambda previous: 3 + len(previous),
        max_length=100,
    )

    assert selected.previous_sentences == ("first sentence", "second sentence")
    assert selected.previous_text == "first sentence\nsecond sentence"


def test_overlong_target_forces_empty_context_and_records_truncation():
    selected = select_complete_context(["one", "two"], lambda _previous: 300, max_length=256)

    assert selected.previous_sentences == ()
    assert selected.input_token_count == 256
    assert selected.target_token_count == 300
    assert selected.target_truncated is True


def test_shared_python_rust_context_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "emotion-context-policy-v2.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in cases:
        counts = case["tokenCounts"]
        selected = select_complete_context(
            case["previousSentences"],
            lambda previous_text: counts[previous_text],
            max_length=case["maxLength"],
        )
        assert selected.previous_text == case["expectedPreviousText"], case["name"]
        assert len(selected.previous_sentences) == case["expectedSentenceCount"], case["name"]
        assert selected.input_token_count == case["expectedInputTokenCount"], case["name"]
        assert selected.target_token_count == case["expectedTargetTokenCount"], case["name"]
        assert selected.context_limited is case["expectedContextLimited"], case["name"]
        assert selected.target_truncated is case["expectedTargetTruncated"], case["name"]


class _CharacterTokenizer:
    def __call__(
        self,
        previous_text,
        current_text,
        *,
        add_special_tokens,
        truncation,
        padding,
        verbose,
    ):
        assert add_special_tokens is True
        assert truncation is False
        assert padding is False
        assert verbose is False
        return {"input_ids": list(range(3 + len(previous_text) + len(current_text)))}


def _record():
    return {
        "schema": "readest-emotion-continuous-v3",
        "schemaVersion": 3,
        "id": "emotion-candidate-book-r000001-s000003",
        "split": "train",
        "workId": "book",
        "previousText": "旧的单句。",
        "text": "目标句。",
        "sentenceType": "narration",
        "sourceKind": "novel",
        "source": "speaker-id:book_annotated",
        "licenseId": "PROPRIETARY-AUTHORIZED",
        "licenseStatus": "test-only",
        "emotions": {
            "happy": 0.01,
            "angry": 0.02,
            "sad": 0.03,
            "afraid": 0.04,
            "disgusted": 0.05,
            "melancholic": 0.06,
            "surprised": 0.07,
            "calm": 0.08,
        },
    }


def test_expand_record_uses_exact_sentence_id_with_repeated_text_and_preserves_labels():
    source_row = {
        "sentences": [
            {"sentenceId": 1, "text": "重复。", "type": "narration"},
            {"sentenceId": 2, "text": "重复。", "type": "narration"},
            {"sentenceId": 3, "text": "目标句。", "type": "narration"},
        ]
    }
    record = _record()

    output, audit = _expand_record(record, source_row, _CharacterTokenizer(), max_length=256)

    assert output["previousText"] == "重复。\n重复。"
    assert output["emotions"] == record["emotions"]
    assert audit["previousSentenceIds"] == [1, 2]
    assert audit["contextSentenceCount"] == 2
    assert audit["targetTruncated"] is False


class _PreflightTokenizer:
    def __call__(self, first, second, **_kwargs):
        return {"input_ids": list(range(3 + len(first) + len(second)))}


def test_training_preflight_detects_context_truncation_but_allows_target_only_truncation():
    from indextts.emotion.schema import EmotionExample

    def example(example_id: str, previous: str, text: str) -> EmotionExample:
        return EmotionExample(
            example_id=example_id,
            work_id="book",
            previous_text=previous,
            text=text,
            sentence_type="narration",
            labels=(0.0,) * 8,
            label_mask=(1.0,) * 8,
            intensity=0.0,
            license_id="PROPRIETARY-AUTHORIZED",
            source="source",
        )

    summary = _complete_context_summary(
        _PreflightTokenizer(),
        [example("context", "a\nb", "target"), example("target", "", "x" * 30)],
        max_length=15,
    )

    assert summary["contextTruncationIds"] == ["context"]
    assert summary["targetTruncationIds"] == ["target"]


def test_training_loader_accepts_continuous_v3_without_legacy_label_fields():
    from indextts.emotion.schema import parse_example

    record = _record()
    example = parse_example(record)

    assert example.labels == tuple(record["emotions"].values())
    assert example.label_mask == (1.0,) * 8
    assert example.intensity == 0.08


def test_python_onnx_runtime_applies_complete_sentence_budget(tmp_path):
    import numpy as np

    from indextts.runtime.emotion import OnnxEmotionProvider

    class Tokenizer:
        final_previous = None

        def __call__(self, first, second, **kwargs):
            if isinstance(first, list):
                self.final_previous = first[0]
                length = 12
                return {
                    "input_ids": np.zeros((1, length), dtype=np.int64),
                    "attention_mask": np.ones((1, length), dtype=np.int64),
                    "token_type_ids": np.zeros((1, length), dtype=np.int64),
                }
            counts = {
                "": 5,
                "near": 8,
                "middle\nnear": 12,
                "old\nmiddle\nnear": 300,
            }
            return {"input_ids": list(range(counts[first]))}

    class Session:
        def run(self, _outputs, _inputs):
            return (
                np.asarray([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32),
                np.asarray([[0.3]], dtype=np.float32),
            )

    tokenizer = Tokenizer()
    provider = OnnxEmotionProvider(tmp_path)
    provider._tokenizer = tokenizer
    provider._session = Session()
    provider.max_length = 256
    provider.neutral_threshold = 0.15

    vector = provider.analyze_context(["old", "middle", "near"], " target\ttext ")

    assert tokenizer.final_previous == "middle\nnear"
    assert vector[0] == pytest.approx(0.2)
