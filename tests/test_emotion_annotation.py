import json

import pytest

from indextts.emotion.annotation import (
    AnnotationValidationError,
    annotation_tree_status,
    merge_annotation_tree,
    merge_double_annotations,
    split_candidates,
    validate_annotation_file,
)
from indextts.emotion.review import resolve_review_queue
from indextts.emotion.source_extract import extract_candidates, sha256_text


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_extract_candidates_is_deterministic_and_hides_speaker_labels(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    _write_jsonl(
        source / "测试书_annotated.jsonl",
        [
            {
                "id": "s0000",
                "sentences": [
                    {"sentenceId": 1, "text": "夜色渐深。", "type": "narration", "lineIndex": 1},
                    {"sentenceId": 2, "text": "“我不答应。”", "type": "dialogue", "lineIndex": 2},
                    {"sentenceId": 3, "text": "他转身离开。", "type": "narration", "lineIndex": 3},
                ],
                "targets": [2],
                "expected": [{"sentenceId": 2, "speakerSlotId": 0}],
            },
            {
                "id": "s0001",
                "sentences": [
                    {"sentenceId": 2, "text": "“我不答应。”", "type": "dialogue", "lineIndex": 2},
                ],
                "targets": [2],
                "expected": [{"sentenceId": 2, "speakerSlotId": 1}],
            },
        ],
    )
    first = extract_candidates(source, tmp_path / "out-a", pool_size=100, seed=7)
    second = extract_candidates(source, tmp_path / "out-b", pool_size=100, seed=7)
    assert first["candidateCount"] == second["candidateCount"] == 3
    assert first["workStats"]["测试书"]["total"] == 3
    assert "difficultyHeuristicCounts" in first
    rows = [json.loads(line) for line in (tmp_path / "out-a" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["sentenceType"] for row in rows} == {"dialogue", "narration"}
    assert all("expected" not in row and "speaker" not in row for row in rows)
    dialogue = next(row for row in rows if row["sentenceType"] == "dialogue")
    assert dialogue["previousText"] == "夜色渐深。"
    assert dialogue["licenseStatus"] == "pending"
    assert dialogue["textSha256"] == sha256_text(dialogue["text"])
    assert (
        (tmp_path / "out-a" / "candidates.jsonl").read_bytes()
        == (tmp_path / "out-b" / "candidates.jsonl").read_bytes()
    )


def _candidate(candidate_id="emotion-candidate-test-r000001-s000001"):
    text = "他说：好。"
    previous = "夜色渐深。"
    return {
        "schema": "readest-emotion-candidate-v1",
        "id": candidate_id,
        "workId": "测试书",
        "text": text,
        "previousText": previous,
        "sentenceType": "dialogue",
        "textSha256": sha256_text(text),
        "previousTextSha256": sha256_text(previous),
        "licenseStatus": "pending",
    }


def _annotation(candidate, *, angry=0, status="accepted", primary="base"):
    return {
        "id": candidate["id"],
        "textSha256": candidate["textSha256"],
        "previousTextSha256": candidate["previousTextSha256"],
        "status": status,
        "emotions": {
            "happy": 0,
            "angry": angry,
            "sad": 0,
            "afraid": 0,
            "disgusted": 0,
            "melancholic": 0,
            "surprised": 0,
            "calm": 0,
        },
        "intensity": 0 if not angry else angry,
        "primaryEmotion": primary if angry else "base",
        "confidence": "high",
        "needsReview": False,
        "hardCases": [],
        "rationale": "测试",
    }


def test_annotation_validation_and_double_merge(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [candidate])
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [_annotation(candidate, angry=0)])
    _write_jsonl(b, [_annotation(candidate, angry=0)])
    report = validate_annotation_file(candidates, a)
    assert report["acceptedCount"] == 1
    merged = merge_double_annotations(candidates, a, b, tmp_path / "merged")
    assert merged["acceptedPendingLicense"] == 1
    assert merged["reviewCount"] == 0
    assert (tmp_path / "merged" / "accepted-pending-license.jsonl").is_file()


def test_annotation_conflict_goes_to_review_and_row_mismatch_is_rejected(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [candidate])
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [_annotation(candidate, angry=0)])
    _write_jsonl(b, [_annotation(candidate, angry=1, primary="angry")])
    merged = merge_double_annotations(candidates, a, b, tmp_path / "merged")
    assert merged["acceptedPendingLicense"] == 0
    assert merged["reviewCount"] == 1
    _write_jsonl(b, [])
    with pytest.raises(AnnotationValidationError, match="行数"):
        validate_annotation_file(candidates, b)


def test_split_candidates_records_input_hash(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [_candidate("a"), _candidate("b"), _candidate("c")])
    manifest = split_candidates(candidates, tmp_path / "chunks", chunk_size=2)
    assert manifest["rowCount"] == 3
    assert len(manifest["chunks"]) == 2
    assert all(chunk["sha256"] for chunk in manifest["chunks"])


def test_merge_annotation_tree_requires_all_pairs_and_aggregates(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    rows = [_candidate("a"), _candidate("b")]
    _write_jsonl(candidates, rows)
    chunks_dir = tmp_path / "chunks"
    split_candidates(candidates, chunks_dir, chunk_size=1)
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    for index, row in enumerate(rows):
        candidate_path = chunks_dir / f"chunk-{index:04d}.jsonl"
        _write_jsonl(a_dir / candidate_path.name, [_annotation(row)])
        _write_jsonl(b_dir / candidate_path.name, [_annotation(row)])
    report = merge_annotation_tree(chunks_dir, a_dir, b_dir, tmp_path / "merged")
    assert report["candidateCount"] == 2
    assert report["acceptedPendingLicense"] == 2
    assert report["chunkCount"] == 2
    status = annotation_tree_status(chunks_dir, a_dir, b_dir)
    assert status["completePairs"] == 2
    assert status["pendingPairs"] == 0

    (b_dir / "chunk-0001.jsonl").unlink()
    with pytest.raises(AnnotationValidationError, match="不完整"):
        merge_annotation_tree(chunks_dir, a_dir, b_dir, tmp_path / "merged-again")


def test_annotation_rejects_agent_license_and_local_path_metadata(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [candidate])
    annotation = _annotation(candidate)
    annotation["licenseId"] = "CC0-1.0"
    path = tmp_path / "annotation.jsonl"
    _write_jsonl(path, [annotation])
    with pytest.raises(AnnotationValidationError, match="许可证"):
        validate_annotation_file(candidates, path)
    annotation.pop("licenseId")
    annotation["rationale"] = "C:\\Users\\test"
    _write_jsonl(path, [annotation])
    with pytest.raises(AnnotationValidationError, match="绝对路径"):
        validate_annotation_file(candidates, path)


def test_test_only_merge_marks_non_release_license_status(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [candidate])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, [_annotation(candidate)])
    _write_jsonl(b, [_annotation(candidate)])
    report = merge_double_annotations(
        candidates, a, b, tmp_path / "merged", license_status="test-only"
    )
    assert report["licenseStatus"] == "test-only"
    merged = json.loads(
        (tmp_path / "merged" / "accepted-pending-license.jsonl").read_text(encoding="utf-8")
    )
    assert merged["licenseStatus"] == "test-only"


def test_resolve_review_queue_requires_complete_final_decisions(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [candidate])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(a, [_annotation(candidate, angry=0)])
    _write_jsonl(b, [_annotation(candidate, angry=1, primary="angry")])
    merged_dir = tmp_path / "merged"
    merge_double_annotations(candidates, a, b, merged_dir)
    queue = merged_dir / "review-queue.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    decision = _annotation(candidate, angry=1, primary="angry")
    _write_jsonl(decisions, [decision])
    report = resolve_review_queue(queue, decisions, tmp_path / "reviewed", license_status="test-only")
    assert report["reviewedAccepted"] == 1
    audit = [
        json.loads(line)
        for line in (tmp_path / "reviewed" / "review-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    queue_row = json.loads(queue.read_text(encoding="utf-8"))
    assert audit[0]["annotationA"] == queue_row["annotationA"]
    assert audit[0]["annotationB"] == queue_row["annotationB"]
    assert report["reviewedDrop"] == 0
    reviewed = json.loads(
        (tmp_path / "reviewed" / "reviewed-accepted.jsonl").read_text(encoding="utf-8")
    )
    assert reviewed["reviewStatus"] == "human-reviewed"
    assert reviewed["licenseStatus"] == "test-only"

    decision["status"] = "review"
    _write_jsonl(decisions, [decision])
    with pytest.raises(AnnotationValidationError, match="accepted 或 drop"):
        resolve_review_queue(queue, decisions, tmp_path / "reviewed-again")
