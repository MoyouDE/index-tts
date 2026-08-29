import json

import pytest

from indextts.emotion.annotation import (
    AnnotationValidationError,
    file_sha256,
    merge_double_annotations,
)
from indextts.emotion.schema import EMOTION_NAMES
from indextts.emotion.review import resolve_review_queue
from indextts.emotion.supplement import integrate_reviewed_supplement


def _emotions(**values):
    result = {name: 0.0 for name in EMOTION_NAMES}
    result.update(values)
    return result


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _base(example_id, work_id, emotions):
    return {
        "schemaVersion": 1,
        "id": example_id,
        "workId": work_id,
        "previousText": "上一句",
        "text": f"基础句-{example_id}",
        "sentenceType": "narration",
        "emotions": emotions,
        "labelMask": {name: 1.0 for name in EMOTION_NAMES},
        "intensity": max(emotions.values()),
        "licenseId": "PROPRIETARY-AUTHORIZED",
        "licenseStatus": "test-only",
        "source": f"speaker-id:{work_id}",
        "annotationStatus": "human-reviewed",
    }


def _coverage(example_id, work_id, emotions, *, reviewed=False):
    row = {
        "id": example_id,
        "workId": work_id,
        "previousText": "补充上一句",
        "text": f"补充句-{example_id}",
        "sentenceType": "dialogue",
        "emotions": emotions,
        "intensity": max(emotions.values()),
        "source": f"speaker-id:{work_id}",
    }
    if reviewed:
        row["reviewStatus"] = "human-reviewed"
    return row


def _agent_annotation(candidate, emotions):
    active = [name for name in EMOTION_NAMES if float(emotions[name]) > 0]
    return {
        "schema": "readest-emotion-annotation-v1",
        "id": candidate["id"],
        "textSha256": candidate.get("textSha256", ""),
        "previousTextSha256": candidate.get("previousTextSha256", ""),
        "status": "accepted",
        "emotions": emotions,
        "intensity": max(emotions.values()),
        "primaryEmotion": active[0] if len(active) == 1 else ("mixed" if active else "base"),
        "confidence": "high",
        "needsReview": False,
        "hardCases": [],
        "rationale": "test",
    }


def _fixture(tmp_path):
    base = tmp_path / "base"
    training_seed = {name: 1.0 for name in EMOTION_NAMES if name != "melancholic"}
    _write_jsonl(base / "train.jsonl", [_base("train-1", "work-a", _emotions(**training_seed))])
    _write_jsonl(base / "dev.jsonl", [_base("dev-1", "work-b", _emotions())])
    _write_jsonl(base / "test.jsonl", [_base("test-1", "work-c", _emotions())])
    merged = tmp_path / "merged"
    _write_jsonl(
        merged / "accepted-pending-license.jsonl",
        [_coverage("coverage-train", "work-a", _emotions(melancholic=0.67))],
    )
    _write_jsonl(merged / "review-queue.jsonl", [])
    return base, merged


def test_integrate_reviewed_supplement_preserves_work_splits(tmp_path):
    base, merged = _fixture(tmp_path)
    report = integrate_reviewed_supplement(
        base,
        merged,
        tmp_path / "output",
        minimum_positive=1,
    )

    assert report["trainingStarted"] is False
    assert report["testTrainingReady"] is True
    assert report["supplementSplitCounts"] == {"train": 1, "dev": 0, "test": 0}
    train = [
        json.loads(line)
        for line in (tmp_path / "output" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train[-1]["id"] == "coverage-train"
    assert train[-1]["annotationStatus"] == "coverage-double-accepted"


def test_integrate_reviewed_supplement_requires_complete_review(tmp_path):
    base, merged = _fixture(tmp_path)
    _write_jsonl(
        merged / "review-queue.jsonl",
        [{"candidate": _coverage("coverage-review", "work-a", _emotions(angry=1))}],
    )

    with pytest.raises(AnnotationValidationError, match="未复核"):
        integrate_reviewed_supplement(base, merged, tmp_path / "output")


def test_integrate_reviewed_supplement_validates_review_hash(tmp_path):
    base, merged = _fixture(tmp_path)
    queue = _write_jsonl(
        merged / "review-queue.jsonl",
        [{"candidate": _coverage("coverage-review", "work-a", _emotions(angry=1))}],
    )
    reviewed = tmp_path / "reviewed"
    candidate = _coverage("coverage-review", "work-a", _emotions(angry=1))
    decisions = _write_jsonl(
        merged / "review-decisions.jsonl",
        [_agent_annotation(candidate, candidate["emotions"])],
    )
    resolve_review_queue(queue, decisions, reviewed, license_status="test-only")

    report = integrate_reviewed_supplement(
        base,
        merged,
        tmp_path / "output",
        reviewed_dir=reviewed,
        minimum_positive=1,
    )
    assert report["review"]["reviewedAccepted"] == 1

    (reviewed / "review-report.json").write_text(
        json.dumps({"queueSha256": "0" * 64, "queueCount": 1}),
        encoding="utf-8",
    )
    with pytest.raises(AnnotationValidationError, match="哈希"):
        integrate_reviewed_supplement(
            base,
            merged,
            tmp_path / "output-again",
            reviewed_dir=reviewed,
        )


def test_integrate_reviewed_supplement_reads_local_brighter_export_name(tmp_path):
    base, merged = _fixture(tmp_path)
    brighter = tmp_path / "brighter"
    _write_jsonl(
        brighter / "brighter-chn-train.jsonl",
        [_base("brighter-1", "brighter-train", _emotions(melancholic=1.0))],
    )
    report = integrate_reviewed_supplement(
        base,
        merged,
        tmp_path / "output",
        brighter_dir=brighter,
        minimum_positive=1,
    )
    assert report["brighterTrainCount"] == 1


def test_integrate_reviewed_supplement_reads_complete_chunk_tree(tmp_path):
    base, _ = _fixture(tmp_path)
    merged_root = tmp_path / "coverage" / "merged"
    chunk = merged_root / "chunk-0000"
    candidate = _coverage("coverage-tree", "work-a", _emotions(melancholic=0.67))
    candidate_path = _write_jsonl(tmp_path / "coverage" / "chunks" / "chunk-0000.jsonl", [candidate])
    annotation = _agent_annotation(candidate, candidate["emotions"])
    a_path = _write_jsonl(tmp_path / "coverage" / "annotations-a" / "chunk-0000.jsonl", [annotation])
    b_path = _write_jsonl(tmp_path / "coverage" / "annotations-b" / "chunk-0000.jsonl", [annotation])
    merge_double_annotations(
        candidate_path,
        a_path,
        b_path,
        chunk,
        license_status="test-only",
    )

    report = integrate_reviewed_supplement(
        base,
        merged_root,
        tmp_path / "output-tree",
        minimum_positive=1,
    )
    assert report["review"]["chunks"] == 1
    assert report["supplementSplitCounts"] == {"train": 1, "dev": 0, "test": 0}
    assert "chunk-0000" in report["sourceHashes"]["coverage"]
    assert report["sourceHashes"]["coverage"]["chunk-0000"]["annotationA"] == file_sha256(a_path)

    _write_jsonl(a_path, [_agent_annotation(candidate, _emotions())])
    with pytest.raises(AnnotationValidationError, match="重算"):
        integrate_reviewed_supplement(
            base,
            merged_root,
            tmp_path / "output-tree-tampered",
            minimum_positive=1,
        )


def test_integrate_reviewed_supplement_rejects_incomplete_chunk_tree(tmp_path):
    base, _ = _fixture(tmp_path)
    merged_root = tmp_path / "coverage" / "merged"
    chunk = merged_root / "chunk-0000"
    candidate = _coverage("coverage-tree", "work-a", _emotions(melancholic=0.67))
    candidate_path = _write_jsonl(tmp_path / "coverage" / "chunks" / "chunk-0000.jsonl", [candidate])
    annotation = _agent_annotation(candidate, candidate["emotions"])
    a_path = _write_jsonl(tmp_path / "coverage" / "annotations-a" / "chunk-0000.jsonl", [annotation])
    b_path = _write_jsonl(tmp_path / "coverage" / "annotations-b" / "chunk-0000.jsonl", [annotation])
    merge_double_annotations(
        candidate_path,
        a_path,
        b_path,
        chunk,
        license_status="test-only",
    )
    _write_jsonl(chunk / "audit.jsonl", [])

    with pytest.raises(AnnotationValidationError, match="audit"):
        integrate_reviewed_supplement(base, merged_root, tmp_path / "output-tree")


def test_integrate_reviewed_supplement_deduplicates_exact_context_with_audit(tmp_path):
    base, merged = _fixture(tmp_path)
    duplicate = _coverage("coverage-duplicate", "work-a", _emotions(melancholic=1.0))
    duplicate.update(
        {
            "previousText": "上一句",
            "text": "基础句-train-1",
            "sentenceType": "narration",
        }
    )
    _write_jsonl(merged / "accepted-pending-license.jsonl", [duplicate])

    report = integrate_reviewed_supplement(
        base,
        merged,
        tmp_path / "output-dedup",
        minimum_positive=1,
    )
    assert report["supplementDeduplicated"] == 1
    assert report["supplementSplitCounts"] == {"train": 0, "dev": 0, "test": 0}
    audit = [
        json.loads(line)
        for line in (tmp_path / "output-dedup" / "supplement-dedup-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert audit == [
        {
            "droppedId": "coverage-duplicate",
            "keptId": "train-1",
            "reason": "exact-context-duplicate",
            "textSha256": "",
            "workId": "work-a",
        }
    ]
