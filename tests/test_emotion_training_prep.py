import json

import pytest

from indextts.emotion.annotation import AnnotationValidationError
from indextts.emotion.schema import EMOTION_NAMES
from indextts.emotion.source_extract import sha256_text
from indextts.emotion.training_prep import prepare_training_data
from indextts.emotion.training_preflight import _length_summary


def _emotions(**values):
    result = {name: 0.0 for name in EMOTION_NAMES}
    result.update(values)
    return result


def _example(example_id, work_id, emotions, intensity, *, text=None):
    return {
        "schemaVersion": 1,
        "id": example_id,
        "workId": work_id,
        "previousText": f"{work_id} 的上一句。",
        "text": text or f"{work_id} 的当前句。",
        "sentenceType": "narration",
        "emotions": emotions,
        "labelMask": {name: 1.0 for name in EMOTION_NAMES},
        "intensity": intensity,
        "licenseId": "PROPRIETARY-AUTHORIZED",
        "licenseStatus": "test-only",
        "source": f"speaker-id:{work_id}",
        "annotationStatus": "human-reviewed",
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _adjudication(base, *, qwen_emotions, decision_emotions, confidence, chosen_source):
    intensity = max(decision_emotions.values())
    active = [name for name in EMOTION_NAMES if decision_emotions[name] > 0]
    return {
        "schema": "readest-emotion-adjudication-result-v1",
        "id": base["id"],
        "workId": base["workId"],
        "sentenceType": base["sentenceType"],
        "previousText": base["previousText"],
        "text": base["text"],
        "agentJudgment": {
            "emotions": base["emotions"],
            "intensity": base["intensity"],
            "primaryEmotion": "base"
            if not any(base["emotions"].values())
            else max(base["emotions"], key=base["emotions"].get),
        },
        "qwenJudgment": {
            "emotionsNeutralAdjusted": qwen_emotions,
            "intensity": max(qwen_emotions.values()),
        },
        "severity": {"maxAbsoluteDifference": 1.0},
        "secondDecision": {
            "schema": "readest-emotion-adjudication-v1",
            "id": base["id"],
            "textSha256": sha256_text(base["text"]),
            "previousTextSha256": sha256_text(base["previousText"]),
            "disposition": "keep",
            "emotions": decision_emotions,
            "intensity": intensity,
            "primaryEmotion": "base"
            if not active
            else (active[0] if len(active) == 1 else "mixed"),
            "chosenSource": chosen_source,
            "confidence": confidence,
            "rationale": "测试二次决断",
        },
    }


def _fixture_tree(tmp_path):
    base_dir = tmp_path / "base"
    high = _example("train-high", "work-train", _emotions(happy=1.0), 1.0)
    low = _example("train-low", "work-train", _emotions(), 0.0, text="突然传来怒吼。")
    dev = _example("dev-1", "work-dev", _emotions(calm=0.33), 0.33)
    test = _example("test-1", "work-test", _emotions(sad=0.67), 0.67)
    _write_jsonl(base_dir / "train.jsonl", [high, low])
    _write_jsonl(base_dir / "dev.jsonl", [dev])
    _write_jsonl(base_dir / "test.jsonl", [test])
    adjudications = [
        _adjudication(
            high,
            qwen_emotions=_emotions(),
            decision_emotions=_emotions(),
            confidence="high",
            chosen_source="qwen",
        ),
        _adjudication(
            low,
            qwen_emotions=_emotions(angry=1.0),
            decision_emotions=_emotions(),
            confidence="low",
            chosen_source="agent",
        ),
    ]
    adjudication_path = _write_jsonl(tmp_path / "adjudicated.jsonl", adjudications)
    return base_dir, adjudication_path


def test_prepare_training_data_applies_decisions_and_quarantines_low_confidence(tmp_path):
    base_dir, adjudication_path = _fixture_tree(tmp_path)
    first = prepare_training_data(
        base_dir,
        [adjudication_path],
        tmp_path / "prepared-a",
        expected_adjudications=2,
    )
    second = prepare_training_data(
        base_dir,
        [adjudication_path],
        tmp_path / "prepared-b",
        expected_adjudications=2,
    )

    assert first["trainingStarted"] is False
    assert first["testTrainingReady"] is True
    assert first["formalTrainingReady"] is False
    assert first["changedRows"] == 1
    assert first["quarantinedLowConfidence"] == 1
    assert first["outputSplitCounts"] == {"train": 1, "dev": 1, "test": 1}
    assert first["outputs"]["splits"] == second["outputs"]["splits"]

    train = [
        json.loads(line)
        for line in (tmp_path / "prepared-a" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert train[0]["id"] == "train-high"
    assert train[0]["emotions"] == _emotions()
    assert train[0]["annotationStatus"] == "second-agent-adjudicated"
    quarantine = (tmp_path / "prepared-a" / "review-required-low-confidence.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(quarantine)["id"] == "train-low"


def test_prepare_training_data_rejects_tampered_hash(tmp_path):
    base_dir, adjudication_path = _fixture_tree(tmp_path)
    rows = [json.loads(line) for line in adjudication_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["secondDecision"]["textSha256"] = "0" * 64
    _write_jsonl(adjudication_path, rows)

    with pytest.raises(AnnotationValidationError, match="textSha256"):
        prepare_training_data(base_dir, [adjudication_path], tmp_path / "prepared")


def test_prepare_training_data_rejects_stale_agent_baseline(tmp_path):
    base_dir, adjudication_path = _fixture_tree(tmp_path)
    rows = [json.loads(line) for line in adjudication_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["agentJudgment"]["emotions"] = _emotions(angry=1.0)
    _write_jsonl(adjudication_path, rows)

    with pytest.raises(AnnotationValidationError, match="训练基线不一致"):
        prepare_training_data(base_dir, [adjudication_path], tmp_path / "prepared")


def test_training_preflight_length_summary_reports_truncation():
    assert _length_summary([5, 10, 20, 30], 16) == {
        "rows": 4,
        "min": 5,
        "p50": 20,
        "p95": 30,
        "p99": 30,
        "max": 30,
        "overMaxLength": 2,
        "overMaxLengthRate": 0.5,
    }
