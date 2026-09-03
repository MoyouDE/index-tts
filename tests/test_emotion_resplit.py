import json

import pytest

from indextts.emotion.resplit import resplit_by_work
from indextts.emotion.schema import EMOTION_NAMES, EmotionExample, load_jsonl_examples


def _example(work: str, index: int, emotion: int = 0) -> EmotionExample:
    labels = [0.0] * len(EMOTION_NAMES)
    labels[emotion] = 0.67
    return EmotionExample(
        example_id=f"{work}-{index}",
        work_id=work,
        previous_text="上文",
        text=f"当前 {index}",
        sentence_type="narration",
        labels=tuple(labels),
        label_mask=(1.0,) * len(labels),
        intensity=0.67,
        license_id="PROPRIETARY-AUTHORIZED",
        source="test",
    )


def _write(path, examples):
    path.write_text(
        "".join(json.dumps(item.as_json(), ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )


def test_resplit_by_work_is_deterministic_and_disjoint(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        *[_example("train-work", index, index) for index in range(8)],
        *[_example("dev-work", index, index) for index in range(8)],
        *[_example("test-work", index, index) for index in range(8)],
    ]
    _write(source, reversed(rows))
    first = tmp_path / "first"
    second = tmp_path / "second"
    options = {
        "dev_works": ["dev-work"],
        "test_works": ["test-work"],
        "minimum_dev_positive": 1,
        "minimum_test_positive": 1,
    }
    report = resplit_by_work([source], first, **options)
    resplit_by_work([source], second, **options)

    assert report["splits"]["dev"]["works"] == ["dev-work"]
    assert report["splits"]["test"]["works"] == ["test-work"]
    assert report["splits"]["train"]["works"] == ["train-work"]
    for split in ("train", "dev", "test"):
        assert (first / f"{split}.jsonl").read_bytes() == (
            second / f"{split}.jsonl"
        ).read_bytes()
        assert {item.work_id for item in load_jsonl_examples(first / f"{split}.jsonl")} == {
            f"{split}-work"
        }


def test_resplit_rejects_overlap_and_insufficient_coverage(tmp_path):
    source = tmp_path / "source.jsonl"
    _write(source, [_example("train", 0), _example("dev", 0), _example("test", 0)])
    with pytest.raises(ValueError, match="重复"):
        resplit_by_work(
            [source], tmp_path / "overlap", dev_works=["dev"], test_works=["dev"]
        )
    with pytest.raises(ValueError, match="覆盖不足"):
        resplit_by_work(
            [source],
            tmp_path / "coverage",
            dev_works=["dev"],
            test_works=["test"],
            minimum_dev_positive=1,
            minimum_test_positive=1,
        )
