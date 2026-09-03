import json
from pathlib import Path

import pytest

from indextts.emotion.continuous_v3 import (
    assemble_continuous_v3_review_chunks,
    detect_conflicts,
    direction_sets,
    finalize_continuous_v3,
    infer_brighter_sentence_type,
    prepare_continuous_v3,
    prepare_continuous_v3_comparison_batches,
    prepare_continuous_v3_reviews,
    validate_continuous_v3_reviews,
)
from indextts.emotion.qwen_continuous import CN_EMOTION_NAMES
from indextts.emotion.schema import EMOTION_NAMES


REVIEW_SCHEMA = "readest-emotion-continuous-review-v3"
BRIGHTER_OBSERVED = {
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "surprised",
}


def _emotions(**values):
    result = {name: 0.0 for name in EMOTION_NAMES}
    result.update(values)
    return result


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _legacy_row(
    sample_id,
    work_id,
    text,
    emotions,
    *,
    source,
    sentence_type="narration",
    previous_text="上一句。",
    label_mask=None,
    chosen_source=None,
):
    row = {
        "schemaVersion": 1,
        "id": sample_id,
        "workId": work_id,
        "previousText": previous_text,
        "text": text,
        "sentenceType": sentence_type,
        "emotions": emotions,
        "labelMask": label_mask or {name: 1.0 for name in EMOTION_NAMES},
        "intensity": max(emotions.values()),
        "licenseId": "CC-BY-4.0" if sample_id.startswith("brighter-") else "PROPRIETARY-AUTHORIZED",
        "licenseStatus": "test-only",
        "source": source,
    }
    if chosen_source is not None:
        row["adjudicationChosenSource"] = chosen_source
    return row


def _fixture_inputs(tmp_path):
    novel_dir = tmp_path / "training-ready-v2"
    brighter_dir = tmp_path / "brighter"
    novel_rows = {
        "train": _legacy_row(
            "novel-train",
            "novel-work-train",
            "他终于笑了。",
            _emotions(happy=1.0),
            source="speaker-id:novel-train",
            chosen_source="qwen",
        ),
        "dev": _legacy_row(
            "novel-dev",
            "novel-work-dev",
            "天色渐渐暗了。",
            _emotions(),
            source="speaker-id:novel-dev",
        ),
        "test": _legacy_row(
            "novel-test",
            "novel-work-test",
            "她高兴地挥了挥手。",
            _emotions(happy=0.67),
            source="speaker-id:novel-test",
        ),
    }
    novel_paths = []
    for split, row in novel_rows.items():
        novel_paths.append(_write_jsonl(novel_dir / f"{split}.jsonl", [row]))

    observed_mask = {
        name: 1.0 if name in BRIGHTER_OBSERVED else 0.0 for name in EMOTION_NAMES
    }
    brighter_rows = {
        "train": _legacy_row(
            "brighter-train",
            "brighter-chn-train",
            " “我真的很开心。” ",
            _emotions(happy=2.0 / 3.0),
            source="brighter-dataset@test/train",
            previous_text="",
            label_mask=observed_mask,
        ),
        "dev": _legacy_row(
            "brighter-dev",
            "brighter-chn-dev",
            "这件事令人难过。",
            _emotions(sad=1.0 / 3.0),
            source="brighter-dataset@test/dev",
            previous_text="",
            label_mask=observed_mask,
        ),
        "test": _legacy_row(
            "brighter-test",
            "brighter-chn-test",
            "这是一条普通消息。",
            _emotions(),
            source="brighter-dataset@test/test",
            previous_text="",
            label_mask=observed_mask,
        ),
    }
    brighter_paths = []
    for split, row in brighter_rows.items():
        brighter_paths.append(
            _write_jsonl(brighter_dir / f"brighter-chn-{split}.jsonl", [row])
        )

    audit = _write_jsonl(
        tmp_path / "adjudication-audit.jsonl",
        [
            {
                "schema": "readest-emotion-training-prep-v1",
                "id": "novel-train",
                # 历史审计早于 v2 作品级重切分；当前 split 以 v2 路径为准。
                "split": "test",
                "workId": "novel-work-train",
                "chosenSource": "qwen",
                "originalEmotions": _emotions(sad=0.67),
                "finalEmotions": _emotions(happy=1.0),
            }
        ],
    )
    return novel_paths, brighter_paths, audit


def _prepare_fixture(tmp_path):
    novel_paths, brighter_paths, audit = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "prepared"
    prepare_continuous_v3(novel_paths, brighter_paths, audit, output_dir)
    candidates_path = output_dir / "candidates.jsonl"
    candidates = [
        json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    return candidates_path, candidates


def _write_qwen_annotations(path, candidates):
    values = {
        "novel-train": _emotions(sad=0.6),
        "novel-dev": _emotions(happy=0.8),
        "novel-test": _emotions(sad=0.8),
        "brighter-train": _emotions(happy=0.7),
        "brighter-dev": _emotions(sad=0.4),
        # melancholic 未被 BRIGHTER 标注，不能单独制造方向冲突。
        "brighter-test": _emotions(melancholic=0.8),
    }
    rows = []
    for candidate in candidates:
        emotions = values[candidate["id"]]
        raw_response = json.dumps(
            {cn: emotions[en] for cn, en in zip(CN_EMOTION_NAMES, EMOTION_NAMES)},
            ensure_ascii=False,
        )
        rows.append(
            {
                "schema": "readest-emotion-qwen-raw-v3",
                "schemaVersion": 3,
                "id": candidate["id"],
                "split": candidate["split"],
                "inputSha256": candidate["inputSha256"],
                "textSha256": candidate["textSha256"],
                "modelFingerprint": "model-test",
                "promptFingerprint": "prompt-test",
                "rawResponse": raw_response,
                "emotions": emotions,
                "attempts": [{"attempt": 1, "rawResponse": raw_response, "error": None}],
            }
        )
    return _write_jsonl(path, rows), values


def _review(queue_row, decision, emotions, confidence="high", reason="测试判断"):
    return {
        "schema": REVIEW_SCHEMA,
        "schemaVersion": 3,
        "id": queue_row["id"],
        "inputSha256": queue_row["inputSha256"],
        "decision": decision,
        "emotions": emotions,
        "confidence": confidence,
        "reason": reason,
    }


def test_prepare_continuous_v3_recovers_agent_directions_and_brighter_metadata(tmp_path):
    novel_paths, brighter_paths, audit = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "prepared"
    report = prepare_continuous_v3(novel_paths, brighter_paths, audit, output_dir)
    rows = [
        json.loads(line)
        for line in (output_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert [row["split"] for row in rows].count("train") == 2
    assert report["restoredHistoricalAuditSplitCounts"] == {"test": 1}
    assert report["restoredHistoricalAuditSplitMismatchCount"] == 1

    restored = next(row for row in rows if row["id"] == "novel-train")
    assert restored["oldDirections"] == ["sad"]
    assert restored["oldDominantDirections"] == ["sad"]
    assert "emotions" not in restored
    assert "intensity" not in restored

    brighter = next(row for row in rows if row["id"] == "brighter-train")
    assert set(brighter["observedDirections"]) == BRIGHTER_OBSERVED
    assert brighter["oldDirections"] == ["happy"]
    assert brighter["sentenceType"] == "dialogue"
    assert brighter["sourceKind"] == "brighter"
    assert brighter["inputSha256"] and brighter["textSha256"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "wrong-schema", "schema"),
        ("split", "validation", "split"),
        ("workId", "wrong-work", "workId"),
        ("chosenSource", "revised", "chosenSource"),
        ("finalEmotions", _emotions(happy=0.99), "finalEmotions"),
    ],
)
def test_prepare_continuous_v3_rejects_incompatible_restoration_audit(
    tmp_path, field, value, error
):
    novel_paths, brighter_paths, audit_path = _fixture_inputs(tmp_path)
    audit_row = json.loads(audit_path.read_text(encoding="utf-8").strip())
    audit_row[field] = value
    _write_jsonl(audit_path, [audit_row])

    with pytest.raises(ValueError, match=error):
        prepare_continuous_v3(
            novel_paths,
            brighter_paths,
            audit_path,
            tmp_path / "prepared-invalid",
        )


def test_prepare_continuous_v3_rejects_restoration_audit_id_mismatch(tmp_path):
    novel_paths, brighter_paths, audit_path = _fixture_inputs(tmp_path)
    audit_row = json.loads(audit_path.read_text(encoding="utf-8").strip())
    audit_row["id"] = "wrong-id"
    _write_jsonl(audit_path, [audit_row])

    with pytest.raises(ValueError, match="审计"):
        prepare_continuous_v3(
            novel_paths,
            brighter_paths,
            audit_path,
            tmp_path / "prepared-invalid-id",
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" “整句对白。” ", "dialogue"),
        ("「整句对白。」", "dialogue"),
        ("『整句对白。』", "dialogue"),
        ("‘整句对白。’", "dialogue"),
        ('"整句对白。"', "dialogue"),
        ("'整句对白。'", "dialogue"),
        ("他说：“只有部分被引号包围。”", "narration"),
        ("“引号不配对。", "narration"),
    ],
)
def test_infer_brighter_sentence_type_uses_conservative_paired_quotes(text, expected):
    assert infer_brighter_sentence_type(text) == expected


def test_direction_sets_and_conflicts_respect_threshold_and_brighter_unknown_dimensions():
    significant, dominant = direction_sets(
        _emotions(happy=0.30, sad=0.20, angry=0.099999)
    )
    assert significant == ("happy", "sad")
    assert dominant == ("happy", "sad")

    novel = {
        "sourceKind": "novel",
        "observedDirections": list(EMOTION_NAMES),
        "oldDirections": [],
        "oldDominantDirections": [],
    }
    assert detect_conflicts(novel, _emotions(happy=0.8))

    brighter = {
        "sourceKind": "brighter",
        "observedDirections": list(BRIGHTER_OBSERVED),
        "oldDirections": [],
        "oldDominantDirections": [],
    }
    assert detect_conflicts(brighter, _emotions(melancholic=0.8)) == ()


def test_equal_significant_direction_set_skips_review_despite_dominance_difference():
    candidate = {
        "sourceKind": "novel",
        "observedDirections": list(EMOTION_NAMES),
        "oldDirections": ["happy", "sad"],
        "oldDominantDirections": ["happy", "sad"],
    }
    assert detect_conflicts(candidate, _emotions(happy=0.7, sad=0.2)) == ()


def _review_fixture(tmp_path):
    candidates_path, candidates = _prepare_fixture(tmp_path)
    qwen_path, qwen_values = _write_qwen_annotations(tmp_path / "qwen.jsonl", candidates)
    review_dir = tmp_path / "review-queue"
    prepare_continuous_v3_reviews(candidates_path, qwen_path, review_dir, chunk_size=1)
    queue_path = review_dir / "review-queue.jsonl"
    queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    return candidates_path, candidates, qwen_path, qwen_values, queue_path, queue


def test_review_preparation_excludes_brighter_unknown_dimension_and_chunks(tmp_path):
    *_, queue_path, queue = _review_fixture(tmp_path)
    assert {row["id"] for row in queue} == {"novel-dev", "novel-test"}
    assert all(row["conflictReasons"] for row in queue)
    assert all(row["inputSha256"] and row["candidateInputSha256"] for row in queue)
    chunks = sorted((queue_path.parent / "chunks").glob("chunk-*.jsonl"))
    assert len(chunks) == 2
    assert all(len(path.read_text(encoding="utf-8").splitlines()) == 1 for path in chunks)


def test_comparison_batches_regroup_without_changing_review_rows(tmp_path):
    *_, queue_path, queue = _review_fixture(tmp_path)
    output_dir = tmp_path / "comparison-batches"
    report = prepare_continuous_v3_comparison_batches(
        queue_path,
        output_dir,
        batch_size=2,
    )

    regrouped = [
        json.loads(line)
        for line in (output_dir / "review-queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["reviewCount"] == len(queue) == 2
    assert report["batchSize"] == 2
    assert report["batchCount"] == 1
    assert report["batches"][0]["rows"] == 2
    assert {row["id"] for row in regrouped} == {row["id"] for row in queue}
    assert {row["id"]: row["inputSha256"] for row in regrouped} == {
        row["id"]: row["inputSha256"] for row in queue
    }
    assert {row["id"]: row for row in regrouped} == {row["id"]: row for row in queue}
    assert report["batches"][0]["primaryDirectionCounts"]
    assert report["batches"][0]["anchorDirectionSignatures"]


def test_comparison_batches_require_more_than_one_row_per_batch(tmp_path):
    *_, queue_path, _ = _review_fixture(tmp_path)
    with pytest.raises(ValueError, match="大于 1"):
        prepare_continuous_v3_comparison_batches(queue_path, tmp_path / "invalid", batch_size=1)


def test_assemble_review_chunks_requires_exact_complete_validated_set(tmp_path):
    *_, queue_path, queue = _review_fixture(tmp_path)
    chunks_dir = tmp_path / "single-pass-chunks"
    for index, row in enumerate(queue):
        _write_jsonl(
            chunks_dir / f"chunk-{index:04d}.jsonl",
            [_review(row, "keep", row["qwenEmotions"], "high")],
        )
    output = tmp_path / "single-pass.jsonl"
    report = assemble_continuous_v3_review_chunks(
        queue_path,
        chunks_dir,
        output,
        pass_name="single-pass",
        chunk_size=1,
    )
    assert report["reviewCount"] == 2
    assert report["chunkCount"] == 2
    assert report["keepCount"] == 2
    assembled = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in assembled] == [row["id"] for row in queue]

    (chunks_dir / "chunk-0001.jsonl").unlink()
    with pytest.raises(ValueError, match="分片集合不完整"):
        assemble_continuous_v3_review_chunks(
            queue_path,
            chunks_dir,
            output,
            pass_name="single-pass",
            chunk_size=1,
        )


def test_single_review_validation_preserves_continuous_values_and_blocks_invalid_keep(tmp_path):
    _, _, _, qwen_values, queue_path, queue = _review_fixture(tmp_path)
    by_id = {row["id"]: row for row in queue}
    reviews = [
        _review(by_id["novel-dev"], "revise", _emotions(happy=0.537), "high"),
        _review(by_id["novel-test"], "keep", qwen_values["novel-test"], "high"),
    ]
    reviews_path = _write_jsonl(tmp_path / "reviews.jsonl", reviews)
    validated = validate_continuous_v3_reviews(queue_path, reviews_path, "single-pass")
    assert validated["reviewCount"] == 2
    assert validated["reviews"][0]["emotions"]["happy"] == pytest.approx(0.537)

    invalid = dict(reviews[1])
    invalid["emotions"] = _emotions(happy=0.8)
    invalid_path = _write_jsonl(tmp_path / "invalid-review.jsonl", [reviews[0], invalid])
    with pytest.raises(ValueError, match="keep|Qwen|向量"):
        validate_continuous_v3_reviews(queue_path, invalid_path, "invalid")


def test_finalize_continuous_v3_writes_only_continuous_label_schema_and_audit(tmp_path):
    candidates_path, candidates, qwen_path, qwen_values, queue_path, queue = _review_fixture(
        tmp_path
    )
    by_id = {row["id"]: row for row in queue}
    reviews_path = _write_jsonl(
        tmp_path / "agent-reviews.jsonl",
        [
            _review(by_id["novel-dev"], "revise", _emotions(), "high"),
            _review(by_id["novel-test"], "keep", qwen_values["novel-test"], "high"),
        ],
    )

    final_dir = tmp_path / "training-ready-v3"
    finalize_continuous_v3(
        candidates_path,
        qwen_path,
        reviews_path,
        final_dir,
        expected_split_counts={"train": 2, "dev": 2, "test": 2},
    )
    final_rows = []
    for split in ("train", "dev", "test"):
        rows = [
            json.loads(line)
            for line in (final_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 2
        assert all(row["split"] == split for row in rows)
        final_rows.extend(rows)
    assert len({row["id"] for row in final_rows}) == len(candidates) == 6
    for row in final_rows:
        assert row["schema"] == "readest-emotion-continuous-v3"
        assert row["schemaVersion"] == 3
        assert tuple(row["emotions"]) == EMOTION_NAMES
        assert all(0.0 <= value <= 1.0 for value in row["emotions"].values())
        assert not {"intensity", "primaryEmotion", "labelMask"}.intersection(row)
    # Agent 明确修订为 base 时必须保留严格全零，不能被改成 calm=1。
    novel_dev = next(row for row in final_rows if row["id"] == "novel-dev")
    assert not any(novel_dev["emotions"].values())

    assert (final_dir / "manifest.json").is_file()
    assert (final_dir / "report.json").is_file()
    for name in (
        "candidates.jsonl",
        "qwen-annotations.jsonl",
        "review-queue.jsonl",
        "agent-reviews.jsonl",
        "final-sources.jsonl",
    ):
        assert (final_dir / "audit" / name).is_file()
    expected_queue = (queue_path).read_text(encoding="utf-8")
    final_queue = (final_dir / "audit" / "review-queue.jsonl").read_text(encoding="utf-8")
    assert final_queue == expected_queue
    report = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    queue_hash = report["fileSha256"]["audit/review-queue.jsonl"]
    assert manifest["fileSha256"]["audit/review-queue.jsonl"] == queue_hash


def test_finalize_continuous_v3_accepts_one_agent_pass_directly(tmp_path):
    candidates_path, _, qwen_path, qwen_values, queue_path, queue = _review_fixture(tmp_path)
    by_id = {row["id"]: row for row in queue}
    reviews_path = _write_jsonl(
        tmp_path / "agent-reviews.jsonl",
        [
            _review(by_id["novel-dev"], "revise", _emotions(happy=0.37), "high"),
            _review(by_id["novel-test"], "keep", qwen_values["novel-test"], "medium"),
        ],
    )

    final_dir = tmp_path / "training-ready-single-pass-v3"
    report = finalize_continuous_v3(
        candidates_path,
        qwen_path,
        reviews_path,
        final_dir,
        expected_split_counts={"train": 2, "dev": 2, "test": 2},
    )

    dev_rows = [
        json.loads(line)
        for line in (final_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    revised = next(row for row in dev_rows if row["id"] == "novel-dev")
    assert revised["emotions"]["happy"] == pytest.approx(0.37)
    assert report["review"]["mode"] == "single-agent"
    assert report["review"]["resolutionCounts"] == {"single-agent": 2}
    audited_reviews = [
        json.loads(line)
        for line in (final_dir / "audit" / "agent-reviews.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    source_reviews = [
        json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines()
    ]
    assert audited_reviews == source_reviews
    assert (final_dir / "audit" / "review-queue.jsonl").read_text(
        encoding="utf-8"
    ) == queue_path.read_text(encoding="utf-8")
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reviewMode"] == "single-agent"
    assert "mergedReviewFileSha256" not in manifest
