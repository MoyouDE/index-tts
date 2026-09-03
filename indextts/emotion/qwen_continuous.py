"""Strict, resumable raw-Qwen labelling for continuous emotion data v3.

The deployed IndexTTS Qwen model was fine-tuned with one system instruction and
the target text as the user message. This module intentionally keeps that
prompt shape. It does *not* call :class:`QwenEmotion.convert`, because that
wrapper fills missing keys, clamps values, and turns an all-zero answer into
``calm=1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .annotation import file_sha256
from .schema import EMOTION_NAMES


QWEN_RAW_SCHEMA = "readest-emotion-qwen-raw-v3"
CANDIDATE_SCHEMA = "readest-emotion-continuous-candidate-v3"
CN_EMOTION_NAMES = (
    "高兴",
    "愤怒",
    "悲伤",
    "恐惧",
    "反感",
    "低落",
    "惊讶",
    "自然",
)
CN_TO_EN = dict(zip(CN_EMOTION_NAMES, EMOTION_NAMES))
PROMPT_SPEC = {
    "system": "文本情感分类",
    "user": "TARGET_TEXT_ONLY",
    "enableThinking": False,
    "doSample": False,
}
MAX_ATTEMPTS = 3
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RawQwenValidationError(ValueError):
    """Raised when a raw generation is not an exact eight-value JSON object."""


def _reject_json_constant(value: str) -> None:
    raise RawQwenValidationError(f"不允许非有限 JSON 数值: {value}")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RawQwenValidationError(f"JSON 对象包含重复键: {key}")
        result[key] = value
    return result


def strict_parse_raw_emotions(content: str) -> dict[str, float]:
    """Parse an exact raw Qwen response without semantic post-processing."""
    if not isinstance(content, str):
        raise RawQwenValidationError("Qwen 输出必须是字符串")
    try:
        payload = json.loads(
            content,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (json.JSONDecodeError, RawQwenValidationError) as exc:
        raise RawQwenValidationError(f"Qwen 输出不是严格 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RawQwenValidationError("Qwen 输出必须是 JSON 对象")
    actual = set(payload)
    expected = set(CN_EMOTION_NAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RawQwenValidationError(f"八维键不完整，缺失={missing}，多余={extra}")
    result: dict[str, float] = {}
    for cn_name in CN_EMOTION_NAMES:
        value = payload[cn_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RawQwenValidationError(f"{cn_name} 必须是 JSON 数值")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RawQwenValidationError(f"{cn_name} 必须是 [0,1] 有限数值")
        # “自然” is the model's eighth native output, hence maps directly to
        # calm. In particular, an all-zero response stays all-zero.
        result[CN_TO_EN[cn_name]] = number
    return result


def _canonical_hash(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def prompt_fingerprint() -> str:
    return _canonical_hash(PROMPT_SPEC)


def model_directory_fingerprint(model_dir: str | Path) -> str:
    """Hash every regular model file so resume cannot silently change weights."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Qwen 模型目录不存在: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Qwen 模型目录为空: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: object, field: str, sample_id: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"候选 {sample_id} 的 {field} 不是小写 SHA-256")
    return text


def _read_candidates(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"候选 JSONL 解析失败 {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"候选第 {line_number} 行不是对象")
            sample_id_value = row.get("id")
            if not isinstance(sample_id_value, str) or not sample_id_value.strip():
                raise ValueError(f"候选 ID 缺失: 第 {line_number} 行")
            sample_id = sample_id_value
            if sample_id in seen:
                raise ValueError(f"候选 ID 重复: {sample_id}")
            if row.get("schema") != CANDIDATE_SCHEMA:
                raise ValueError(f"候选 schema 无效: {sample_id}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"候选目标句为空: {sample_id}")
            if row.get("split") not in {"train", "dev", "test"}:
                raise ValueError(f"候选 split 无效: {sample_id}")
            _validate_sha256(row.get("inputSha256"), "inputSha256", sample_id)
            text_sha = _validate_sha256(row.get("textSha256"), "textSha256", sample_id)
            expected_text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_sha != expected_text_sha:
                raise ValueError(f"候选 {sample_id} 的 textSha256 与目标句不匹配")
            seen.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"候选文件为空: {path}")
    return rows


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            # Do not sort keys: the v3 interface deliberately emits emotion
            # dimensions in EMOTION_NAMES order.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _part_path(parts_dir: Path, sample_id: str) -> Path:
    safe_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return parts_dir / f"{safe_name}.json"


def _checkpoint_base(
    candidate: Mapping[str, object],
    *,
    model_fingerprint: str,
    prompt_sha256: str,
) -> dict[str, object]:
    return {
        "schema": QWEN_RAW_SCHEMA,
        "schemaVersion": 3,
        "id": candidate["id"],
        "split": candidate["split"],
        "inputSha256": candidate["inputSha256"],
        "textSha256": candidate["textSha256"],
        "candidateFingerprint": _canonical_hash(candidate),
        "modelFingerprint": model_fingerprint,
        "promptFingerprint": prompt_sha256,
        "status": "pending",
        "attempts": [],
    }


def _normalize_emotions(value: object, sample_id: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(EMOTION_NAMES):
        raise ValueError(f"Qwen 检查点八维键无效: {sample_id}")
    # Reuse the raw parser for bool, finite-number and range validation, then
    # restore the public order even if an internal JSON file sorted its keys.
    raw = json.dumps(
        {cn: value[en] for cn, en in zip(CN_EMOTION_NAMES, EMOTION_NAMES)},
        ensure_ascii=False,
    )
    return strict_parse_raw_emotions(raw)


def _validate_checkpoint(
    path: Path,
    candidate: Mapping[str, object],
    *,
    model_fingerprint: str,
    prompt_sha256: str,
    max_attempts: int,
) -> dict[str, object]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qwen 检查点损坏: {path}: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"Qwen 检查点不是对象: {path}")
    sample_id = str(candidate["id"])
    expected = {
        "schema": QWEN_RAW_SCHEMA,
        "id": candidate["id"],
        "split": candidate["split"],
        "inputSha256": candidate["inputSha256"],
        "textSha256": candidate["textSha256"],
        "candidateFingerprint": _canonical_hash(candidate),
        "modelFingerprint": model_fingerprint,
        "promptFingerprint": prompt_sha256,
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise ValueError(f"Qwen 检查点 {sample_id} 的 {field} 指纹不匹配")
    attempts = row.get("attempts")
    if not isinstance(attempts, list) or len(attempts) > max_attempts:
        raise ValueError(f"Qwen 检查点 {sample_id} 的重试记录无效")
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != {
            "attempt",
            "rawResponse",
            "error",
        }:
            raise ValueError(f"Qwen 检查点 {sample_id} 的第 {index} 次记录无效")
        if attempt["attempt"] != index or not isinstance(attempt["rawResponse"], str):
            raise ValueError(f"Qwen 检查点 {sample_id} 的第 {index} 次序号无效")
        error = attempt["error"]
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError(f"Qwen 检查点 {sample_id} 的第 {index} 次错误无效")
        if index < len(attempts) and error is None:
            raise ValueError(f"Qwen 检查点 {sample_id} 在成功后仍有重试")

    status = row.get("status")
    if status == "pending":
        if any(attempt["error"] is None for attempt in attempts):
            raise ValueError(f"Qwen 检查点 {sample_id} 的 pending 状态无效")
        if "emotions" in row or "rawResponse" in row:
            raise ValueError(f"Qwen 检查点 {sample_id} 的 pending 结果不应含标签")
    elif status == "completed":
        if not attempts or attempts[-1]["error"] is not None:
            raise ValueError(f"Qwen 检查点 {sample_id} 的 completed 状态无效")
        raw_response = row.get("rawResponse")
        if not isinstance(raw_response, str) or raw_response != attempts[-1]["rawResponse"]:
            raise ValueError(f"Qwen 检查点 {sample_id} 的最终原始响应不匹配")
        emotions = _normalize_emotions(row.get("emotions"), sample_id)
        if strict_parse_raw_emotions(raw_response) != emotions:
            raise ValueError(f"Qwen 检查点 {sample_id} 的原始响应与向量不匹配")
        row["emotions"] = emotions
    else:
        raise ValueError(f"Qwen 检查点 {sample_id} 的状态无效")
    return row


def _public_annotation(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {
        key: checkpoint[key]
        for key in (
            "schema",
            "schemaVersion",
            "id",
            "split",
            "inputSha256",
            "textSha256",
            "modelFingerprint",
            "promptFingerprint",
            "rawResponse",
            "emotions",
            "attempts",
        )
    }


def annotate_qwen_continuous(
    candidates_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    max_retries: int = MAX_ATTEMPTS,
    generator: Callable[[Sequence[str]], Sequence[str]] | None = None,
    model_fingerprint: str | None = None,
) -> dict[str, object]:
    """Generate strict labels with a separately atomic checkpoint for each ID."""
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size 和 max_new_tokens 必须大于 0")
    if not 1 <= max_retries <= MAX_ATTEMPTS:
        raise ValueError(f"max_retries 必须位于 [1,{MAX_ATTEMPTS}]")
    candidates = _read_candidates(candidates_path)
    destination = Path(output_dir)
    parts_dir = destination / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    model_sha = model_fingerprint or model_directory_fingerprint(model_dir)
    if not isinstance(model_sha, str) or not model_sha:
        raise ValueError("model_fingerprint 不能为空")
    prompt_sha = prompt_fingerprint()
    run_config = {
        "schema": QWEN_RAW_SCHEMA,
        "candidateFileSha256": file_sha256(candidates_path),
        "candidateCount": len(candidates),
        "modelFingerprint": model_sha,
        "promptFingerprint": prompt_sha,
        "batchSize": batch_size,
        "maxNewTokens": max_new_tokens,
        "maxRetries": max_retries,
    }
    run_fingerprint = _canonical_hash(run_config)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"已有 Qwen manifest 损坏: {exc}") from exc
        if (
            not isinstance(previous, dict)
            or previous.get("runFingerprint") != run_fingerprint
            or previous.get("prompt") != PROMPT_SPEC
            or any(previous.get(key) != value for key, value in run_config.items())
        ):
            raise ValueError("已有 Qwen 续跑目录的候选、输入、模型或提示指纹不匹配")
    _atomic_json(
        manifest_path,
        {
            **run_config,
            "runFingerprint": run_fingerprint,
            "status": "running",
            "prompt": PROMPT_SPEC,
        },
    )

    if generator is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        def generate_texts(texts: Sequence[str]) -> Sequence[str]:
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": PROMPT_SPEC["system"]},
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for text in texts
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = encoded.input_ids.shape[1]
            return tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )

        generator = generate_texts

    states: dict[str, dict[str, object]] = {}
    pending: list[dict[str, object]] = []
    for candidate in candidates:
        sample_id = str(candidate["id"])
        checkpoint_path = _part_path(parts_dir, sample_id)
        if checkpoint_path.is_file():
            state = _validate_checkpoint(
                checkpoint_path,
                candidate,
                model_fingerprint=model_sha,
                prompt_sha256=prompt_sha,
                max_attempts=max_retries,
            )
        else:
            state = _checkpoint_base(
                candidate,
                model_fingerprint=model_sha,
                prompt_sha256=prompt_sha,
            )
        states[sample_id] = state
        attempts = state["attempts"]
        assert isinstance(attempts, list)
        if state["status"] == "pending" and len(attempts) < max_retries:
            pending.append(candidate)

    started_at = time.monotonic()
    while pending:
        next_pending: list[dict[str, object]] = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            texts = [candidate["text"] for candidate in batch]
            outputs = list(generator(texts))
            if len(outputs) != len(batch):
                raise RuntimeError("Qwen generator 返回数量与输入不一致")
            for candidate, output in zip(batch, outputs):
                sample_id = str(candidate["id"])
                state = states[sample_id]
                attempts = state["attempts"]
                assert isinstance(attempts, list)
                raw_response = output if isinstance(output, str) else str(output)
                error: str | None = None
                try:
                    emotions = strict_parse_raw_emotions(raw_response)
                except RawQwenValidationError as exc:
                    emotions = None
                    error = str(exc)
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "rawResponse": raw_response,
                        "error": error,
                    }
                )
                if emotions is None:
                    if len(attempts) < max_retries:
                        next_pending.append(candidate)
                else:
                    state["status"] = "completed"
                    state["rawResponse"] = raw_response
                    state["emotions"] = emotions
                # Persist every response, including failed attempts, before
                # moving to another ID.
                _atomic_json(_part_path(parts_dir, sample_id), state)
            done_count = sum(state["status"] == "completed" for state in states.values())
            elapsed = max(time.monotonic() - started_at, 0.001)
            rate = done_count / elapsed
            eta = (len(candidates) - done_count) / rate if rate else 0.0
            print(
                f"Qwen 连续重标 {done_count}/{len(candidates)} ETA {eta / 60:.1f} min",
                file=sys.stderr,
                flush=True,
            )
        pending = next_pending

    failed = [
        sample_id
        for sample_id, state in states.items()
        if state["status"] != "completed"
    ]
    if failed:
        failure_path = destination / "invalid-responses.json"
        _atomic_json(
            failure_path,
            {
                "failedIds": failed,
                "attempts": {sample_id: states[sample_id]["attempts"] for sample_id in failed},
            },
        )
        raise RuntimeError(f"Qwen 严格输出重试失败，阻止继续: {failed[:5]}")

    completed = [_public_annotation(states[str(candidate["id"])]) for candidate in candidates]
    completed_ids = [str(row["id"]) for row in completed]
    if len(completed) != len(candidates) or len(set(completed_ids)) != len(completed_ids):
        raise RuntimeError("Qwen 汇总数量不完整或包含重复 ID")
    annotations_path = destination / "annotations.jsonl"
    _atomic_jsonl(annotations_path, completed)
    report = {
        **run_config,
        "runFingerprint": run_fingerprint,
        "status": "completed",
        "prompt": PROMPT_SPEC,
        "annotationFile": str(annotations_path.resolve()),
        "annotationFileSha256": file_sha256(annotations_path),
        "completedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(manifest_path, report)
    return report


def materialize_completed_qwen_continuous(
    candidates_path: str | Path,
    qwen_output_dir: str | Path,
) -> dict[str, object]:
    """Materialize only strict successes after an explicitly accepted failed run.

    This does not repair, coerce, or regenerate failed outputs.  It validates the
    original run fingerprint and every per-ID checkpoint, then writes an ordered
    candidate/annotation subset plus an immutable exclusion audit.
    """
    candidates = _read_candidates(candidates_path)
    destination = Path(qwen_output_dir)
    manifest_path = destination / "manifest.json"
    failure_path = destination / "invalid-responses.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Qwen 部分物化缺少审计文件: {exc.filename}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen 部分物化审计 JSON 损坏: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("prompt") != PROMPT_SPEC:
        raise ValueError("Qwen manifest 或提示协议无效")
    run_fields = (
        "schema",
        "candidateFileSha256",
        "candidateCount",
        "modelFingerprint",
        "promptFingerprint",
        "batchSize",
        "maxNewTokens",
        "maxRetries",
    )
    run_config = {field: manifest.get(field) for field in run_fields}
    if manifest.get("runFingerprint") != _canonical_hash(run_config):
        raise ValueError("Qwen manifest 的 runFingerprint 无效")
    if run_config["schema"] != QWEN_RAW_SCHEMA:
        raise ValueError("Qwen manifest schema 无效")
    if run_config["candidateFileSha256"] != file_sha256(candidates_path):
        raise ValueError("Qwen manifest 候选文件哈希不匹配")
    if run_config["candidateCount"] != len(candidates):
        raise ValueError("Qwen manifest 候选数量不匹配")
    if run_config["promptFingerprint"] != prompt_fingerprint():
        raise ValueError("Qwen manifest 提示指纹不匹配")
    model_sha = str(run_config["modelFingerprint"])
    max_attempts = run_config["maxRetries"]
    if not _SHA256.fullmatch(model_sha) or not isinstance(max_attempts, int):
        raise ValueError("Qwen manifest 模型指纹或重试次数无效")

    completed_candidates: list[dict[str, object]] = []
    completed_annotations: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    failed_ids: list[str] = []
    failed_attempts: dict[str, object] = {}
    parts_dir = destination / "parts"
    for candidate in candidates:
        sample_id = str(candidate["id"])
        checkpoint_path = _part_path(parts_dir, sample_id)
        if not checkpoint_path.is_file():
            raise ValueError(f"Qwen 部分物化缺少检查点: {sample_id}")
        state = _validate_checkpoint(
            checkpoint_path,
            candidate,
            model_fingerprint=model_sha,
            prompt_sha256=str(run_config["promptFingerprint"]),
            max_attempts=max_attempts,
        )
        if state["status"] == "completed":
            completed_candidates.append(candidate)
            completed_annotations.append(_public_annotation(state))
            continue
        attempts = state["attempts"]
        assert isinstance(attempts, list)
        if len(attempts) != max_attempts:
            raise ValueError(f"Qwen 失败检查点尚未耗尽重试: {sample_id}")
        failed_ids.append(sample_id)
        failed_attempts[sample_id] = attempts
        exclusions.append(
            {
                "schema": "readest-emotion-qwen-exclusion-v3",
                "schemaVersion": 3,
                "id": sample_id,
                "split": candidate["split"],
                "sourceKind": candidate.get("sourceKind", ""),
                "candidateInputSha256": candidate["inputSha256"],
                "textSha256": candidate["textSha256"],
                "modelFingerprint": model_sha,
                "promptFingerprint": run_config["promptFingerprint"],
                "attempts": attempts,
                "reason": "strict-qwen-output-failed-after-three-attempts",
            }
        )

    if not isinstance(failure, dict) or failure.get("failedIds") != failed_ids:
        raise ValueError("Qwen 失败审计 ID 与检查点不一致")
    if failure.get("attempts") != failed_attempts:
        raise ValueError("Qwen 失败审计重试记录与检查点不一致")
    if not exclusions:
        raise ValueError("Qwen 运行没有失败项，不需要部分物化")

    candidates_valid_path = destination / "candidates-valid.jsonl"
    annotations_valid_path = destination / "annotations-valid.jsonl"
    exclusions_path = destination / "excluded-invalid.jsonl"
    _atomic_jsonl(candidates_valid_path, completed_candidates)
    _atomic_jsonl(annotations_valid_path, completed_annotations)
    _atomic_jsonl(exclusions_path, exclusions)
    report = {
        "schema": "readest-emotion-qwen-partial-materialization-v3",
        "schemaVersion": 3,
        "status": "completed-with-explicit-exclusions",
        "candidateCount": len(candidates),
        "completedCount": len(completed_candidates),
        "excludedCount": len(exclusions),
        "completedSplitCounts": dict(Counter(str(row["split"]) for row in completed_candidates)),
        "excludedSplitCounts": dict(Counter(str(row["split"]) for row in exclusions)),
        "excludedSourceCounts": dict(Counter(str(row["sourceKind"]) for row in exclusions)),
        "runFingerprint": manifest["runFingerprint"],
        "candidateFileSha256": file_sha256(candidates_path),
        "validCandidateFile": str(candidates_valid_path.resolve()),
        "validCandidateFileSha256": file_sha256(candidates_valid_path),
        "validAnnotationFile": str(annotations_valid_path.resolve()),
        "validAnnotationFileSha256": file_sha256(annotations_valid_path),
        "exclusionFile": str(exclusions_path.resolve()),
        "exclusionFileSha256": file_sha256(exclusions_path),
        "failureAuditFileSha256": file_sha256(failure_path),
    }
    _atomic_json(destination / "materialize-valid-report.json", report)
    return report
