import hashlib
import json

import pytest

from indextts.emotion.release import benchmark_qwen_annotations
from indextts.emotion.schema import EMOTION_NAMES, EmotionExample


def _example(example_id="sample", text="当前", previous="上文"):
    return EmotionExample(
        example_id=example_id,
        work_id="work",
        previous_text=previous,
        text=text,
        sentence_type="narration",
        labels=(0.0,) * len(EMOTION_NAMES),
        label_mask=(1.0,) * len(EMOTION_NAMES),
        intensity=0.0,
        license_id="PROPRIETARY-AUTHORIZED",
        source="test",
    )


def _write_annotation(path, example, emotions=None):
    emotions = emotions or {name: 0.0 for name in EMOTION_NAMES}
    row = {
        "id": example.example_id,
        "textSha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
        "previousTextSha256": hashlib.sha256(example.previous_text.encode("utf-8")).hexdigest(),
        "emotions": emotions,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_precomputed_qwen_benchmark_aligns_natural_calm_to_base(tmp_path):
    example = _example()
    path = tmp_path / "qwen.jsonl"
    vector = {name: 0.0 for name in EMOTION_NAMES}
    vector["calm"] = 1.0
    _write_annotation(path, example, vector)

    adjusted = benchmark_qwen_annotations([example], [path])
    raw = benchmark_qwen_annotations([example], [path], neutral_adjust_calm=False)
    assert adjusted["neutralFalseActivationRate"] == 0.0
    assert raw["neutralFalseActivationRate"] == 1.0
    assert adjusted["semanticPolicy"] == "natural-as-base"


def test_precomputed_qwen_benchmark_rejects_missing_or_mismatched_rows(tmp_path):
    example = _example()
    path = tmp_path / "qwen.jsonl"
    _write_annotation(path, example)
    with pytest.raises(ValueError, match="未覆盖"):
        benchmark_qwen_annotations([_example("missing")], [path])

    row = json.loads(path.read_text(encoding="utf-8"))
    row["textSha256"] = "0" * 64
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="textSha256"):
        benchmark_qwen_annotations([example], [path])
