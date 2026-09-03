import hashlib
import json

import pytest

from indextts.emotion.qwen_continuous import (
    CANDIDATE_SCHEMA,
    CN_EMOTION_NAMES,
    PROMPT_SPEC,
    RawQwenValidationError,
    annotate_qwen_continuous,
    materialize_completed_qwen_continuous,
    model_directory_fingerprint,
    strict_parse_raw_emotions,
)
from indextts.emotion.schema import EMOTION_NAMES


MODEL_FINGERPRINT = "a" * 64


def _raw(values=None, *, whitespace=False):
    payload = {name: 0.0 for name in CN_EMOTION_NAMES}
    if values:
        payload.update(values)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f" \n{text}\t" if whitespace else text


def _candidate(sample_id, text, *, split="train"):
    return {
        "schema": CANDIDATE_SCHEMA,
        "id": sample_id,
        "split": split,
        "text": text,
        "previousText": "这段上文绝不能发送给 Qwen",
        "sentenceType": "dialogue",
        "inputSha256": hashlib.sha256(f"{sample_id}\0{text}".encode()).hexdigest(),
        "textSha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _write_candidates(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _annotate(candidates_path, output_dir, generator, **kwargs):
    return annotate_qwen_continuous(
        candidates_path,
        output_dir,
        output_dir / "unused-model",
        generator=generator,
        model_fingerprint=MODEL_FINGERPRINT,
        **kwargs,
    )


def test_prompt_and_raw_mapping_preserve_native_continuous_values():
    assert PROMPT_SPEC == {
        "system": "文本情感分类",
        "user": "TARGET_TEXT_ONLY",
        "enableThinking": False,
        "doSample": False,
    }
    values = strict_parse_raw_emotions(
        _raw({"高兴": 0.123456, "悲伤": 0.81, "自然": 0.37})
    )
    assert list(values) == list(EMOTION_NAMES)
    assert values["happy"] == 0.123456
    assert values["sad"] == 0.81
    assert values["calm"] == 0.37


def test_all_zero_does_not_fall_back_to_calm():
    values = strict_parse_raw_emotions(_raw())
    assert values == {name: 0.0 for name in EMOTION_NAMES}
    assert values["calm"] == 0.0


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        json.dumps({name: 0 for name in CN_EMOTION_NAMES[:-1]}, ensure_ascii=False),
        json.dumps(
            {**{name: 0 for name in CN_EMOTION_NAMES}, "平静": 0},
            ensure_ascii=False,
        ),
        _raw({"高兴": True}),
        _raw({"高兴": "0.5"}),
        _raw({"高兴": -0.01}),
        _raw({"高兴": 1.01}),
        _raw().replace('"高兴":0.0', '"高兴":NaN'),
        _raw().replace('"高兴":0.0', '"高兴":Infinity'),
        _raw().replace('"高兴":0.0', '"高兴":0,"高兴":0.1'),
    ],
)
def test_strict_parser_rejects_malformed_responses(content):
    with pytest.raises(RawQwenValidationError):
        strict_parse_raw_emotions(content)


def test_generator_receives_only_targets_and_annotations_keep_candidate_order(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    rows = [_candidate("b", "第二句", split="dev"), _candidate("a", "第一句")]
    _write_candidates(candidates, rows)
    successful_b = _raw({"自然": 0.4})
    successful_a = _raw({"高兴": 0.234567}, whitespace=True)
    response_batches = iter([["invalid", successful_a], [successful_b]])
    calls = []

    def generator(texts):
        calls.append(list(texts))
        return next(response_batches)

    report = _annotate(candidates, tmp_path / "output", generator, batch_size=2)
    annotations = _read_jsonl(tmp_path / "output" / "annotations.jsonl")

    assert calls == [["第二句", "第一句"], ["第二句"]]
    assert [row["id"] for row in annotations] == ["b", "a"]
    assert list(annotations[0]["emotions"]) == list(EMOTION_NAMES)
    assert annotations[0]["attempts"][0]["rawResponse"] == "invalid"
    assert annotations[0]["attempts"][0]["error"]
    assert annotations[0]["attempts"][1]["rawResponse"] == successful_b
    assert annotations[1]["rawResponse"] == successful_a
    assert annotations[1]["emotions"]["happy"] == 0.234567
    assert report["status"] == "completed"
    assert report["prompt"] == PROMPT_SPEC


def test_failed_response_is_attempted_at_most_three_times_and_audited(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("bad", "无法解析")])
    calls = []

    def generator(texts):
        calls.append(list(texts))
        return ["invalid"]

    with pytest.raises(RuntimeError, match="重试失败"):
        _annotate(candidates, tmp_path / "output", generator)
    assert len(calls) == 3
    failure = json.loads(
        (tmp_path / "output" / "invalid-responses.json").read_text(encoding="utf-8")
    )
    assert failure["failedIds"] == ["bad"]
    assert [item["attempt"] for item in failure["attempts"]["bad"]] == [1, 2, 3]
    assert all(item["rawResponse"] == "invalid" for item in failure["attempts"]["bad"])

    with pytest.raises(ValueError, match=r"\[1,3\]"):
        _annotate(candidates, tmp_path / "other", generator, max_retries=4)

    resumed_calls = []
    with pytest.raises(RuntimeError, match="重试失败"):
        _annotate(
            candidates,
            tmp_path / "output",
            lambda texts: resumed_calls.append(list(texts)) or [_raw()],
        )
    assert resumed_calls == []


def test_materialize_completed_subset_keeps_strict_successes_and_audits_exclusions(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    rows = [
        _candidate("good", "合法目标句", split="dev"),
        _candidate("bad", "非法目标句", split="test"),
    ]
    rows[0]["sourceKind"] = "novel"
    rows[1]["sourceKind"] = "brighter"
    _write_candidates(candidates, rows)
    calls = 0

    def generator(texts):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_raw({"高兴": 0.123456}), "invalid"]
        return ["invalid"]

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="重试失败"):
        _annotate(candidates, output, generator, batch_size=2)

    report = materialize_completed_qwen_continuous(candidates, output)
    valid_candidates = _read_jsonl(output / "candidates-valid.jsonl")
    valid_annotations = _read_jsonl(output / "annotations-valid.jsonl")
    exclusions = _read_jsonl(output / "excluded-invalid.jsonl")

    assert report["completedCount"] == 1
    assert report["excludedCount"] == 1
    assert report["completedSplitCounts"] == {"dev": 1}
    assert report["excludedSplitCounts"] == {"test": 1}
    assert report["excludedSourceCounts"] == {"brighter": 1}
    assert [row["id"] for row in valid_candidates] == ["good"]
    assert [row["id"] for row in valid_annotations] == ["good"]
    assert valid_annotations[0]["emotions"]["happy"] == 0.123456
    assert [row["id"] for row in exclusions] == ["bad"]
    assert len(exclusions[0]["attempts"]) == 3
    assert exclusions[0]["reason"] == "strict-qwen-output-failed-after-three-attempts"


def test_resume_skips_completed_ids_and_preserves_candidate_order(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("one", "甲"), _candidate("two", "乙")])
    first_calls = 0

    def interrupted_generator(texts):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            return [_raw({"高兴": 0.2})]
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _annotate(
            candidates,
            tmp_path / "output",
            interrupted_generator,
            batch_size=1,
        )

    resumed_calls = []

    def resumed_generator(texts):
        resumed_calls.append(list(texts))
        return [_raw({"悲伤": 0.6})]

    _annotate(candidates, tmp_path / "output", resumed_generator, batch_size=1)
    annotations = _read_jsonl(tmp_path / "output" / "annotations.jsonl")
    assert resumed_calls == [["乙"]]
    assert [row["id"] for row in annotations] == ["one", "two"]
    assert annotations[0]["emotions"]["happy"] == 0.2
    assert len(annotations[0]["attempts"]) == 1


def test_resume_fingerprint_includes_batch_size(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("one", "甲")])
    output = tmp_path / "output"

    first = _annotate(candidates, output, lambda texts: [_raw()], batch_size=2)
    resumed_calls = []
    resumed = _annotate(
        candidates,
        output,
        lambda texts: resumed_calls.append(list(texts)) or [_raw()],
        batch_size=2,
    )

    assert resumed_calls == []
    assert first["batchSize"] == resumed["batchSize"] == 2
    assert first["runFingerprint"] == resumed["runFingerprint"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["batchSize"] == 2

    with pytest.raises(ValueError, match="指纹不匹配"):
        _annotate(candidates, output, lambda texts: [_raw()], batch_size=1)


def test_resume_preserves_an_invalid_attempt_before_interruption(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("one", "甲")])
    call_count = 0

    def interrupted_generator(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["invalid-first-response"]
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _annotate(candidates, tmp_path / "output", interrupted_generator)

    _annotate(candidates, tmp_path / "output", lambda texts: [_raw({"恐惧": 0.45})])
    annotation = _read_jsonl(tmp_path / "output" / "annotations.jsonl")[0]
    assert [attempt["attempt"] for attempt in annotation["attempts"]] == [1, 2]
    assert annotation["attempts"][0]["rawResponse"] == "invalid-first-response"
    assert annotation["attempts"][0]["error"]
    assert annotation["attempts"][1]["error"] is None


def test_resume_rejects_candidate_input_and_model_fingerprint_changes(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    row = _candidate("one", "甲")
    _write_candidates(candidates, [row])
    output = tmp_path / "output"
    _annotate(candidates, output, lambda texts: [_raw()])

    changed = dict(row)
    changed["inputSha256"] = "b" * 64
    _write_candidates(candidates, [changed])
    with pytest.raises(ValueError, match="指纹不匹配"):
        _annotate(candidates, output, lambda texts: [_raw()])

    _write_candidates(candidates, [row])
    with pytest.raises(ValueError, match="指纹不匹配"):
        annotate_qwen_continuous(
            candidates,
            output,
            output / "unused-model",
            generator=lambda texts: [_raw()],
            model_fingerprint="b" * 64,
        )


def test_resume_rejects_tampered_prompt_and_candidate_checkpoint_hashes(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("one", "甲")])
    output = tmp_path / "output"
    _annotate(candidates, output, lambda texts: [_raw()])
    part = next((output / "parts").glob("*.json"))
    state = json.loads(part.read_text(encoding="utf-8"))
    state["promptFingerprint"] = "0" * 64
    part.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="promptFingerprint"):
        _annotate(candidates, output, lambda texts: [_raw()])

    state["promptFingerprint"] = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )["promptFingerprint"]
    state["candidateFingerprint"] = "0" * 64
    part.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="candidateFingerprint"):
        _annotate(candidates, output, lambda texts: [_raw()])


def test_candidates_require_unique_ids_and_matching_text_hash(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    duplicate = _candidate("same", "甲")
    _write_candidates(candidates, [duplicate, duplicate])
    with pytest.raises(ValueError, match="重复"):
        _annotate(candidates, tmp_path / "duplicate", lambda texts: [_raw()] * len(texts))

    mismatch = _candidate("one", "甲")
    mismatch["textSha256"] = "0" * 64
    _write_candidates(candidates, [mismatch])
    with pytest.raises(ValueError, match="textSha256"):
        _annotate(candidates, tmp_path / "mismatch", lambda texts: [_raw()])


def test_generator_output_count_must_match_input_count(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    _write_candidates(candidates, [_candidate("one", "甲"), _candidate("two", "乙")])
    with pytest.raises(RuntimeError, match="返回数量"):
        _annotate(candidates, tmp_path / "output", lambda texts: [_raw()], batch_size=2)


def test_model_directory_fingerprint_detects_file_changes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"first")
    first = model_directory_fingerprint(model)
    assert first == model_directory_fingerprint(model)
    weights.write_bytes(b"second")
    assert model_directory_fingerprint(model) != first
