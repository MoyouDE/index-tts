"""Validation, chunking and double-annotation merge for emotion candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping

from .schema import EMOTION_NAMES
from .source_extract import iter_candidate_records, sha256_text


ANNOTATION_SCHEMA = "readest-emotion-annotation-v1"
ALLOWED_LEVELS = (0.0, 0.33, 0.67, 1.0)
ALLOWED_STATUS = frozenset({"accepted", "review", "drop"})
ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})
ALLOWED_PRIMARY = frozenset((*EMOTION_NAMES, "mixed", "base"))


class AnnotationValidationError(ValueError):
    """Raised when an agent result cannot be safely merged."""


_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/])|(?:^[/\\])")


def _reject_agent_metadata(value: object, *, sample_id: str) -> None:
    """Reject fields that could leak local state or forge licensing metadata."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"licenseId", "licenseStatus", "licenseEvidence"}:
                raise AnnotationValidationError(f"{sample_id} 不允许 agent 填写许可证字段")
            _reject_agent_metadata(item, sample_id=sample_id)
    elif isinstance(value, list):
        for item in value:
            _reject_agent_metadata(item, sample_id=sample_id)
    elif isinstance(value, str):
        if _ABSOLUTE_PATH.search(value) or "pickle" in value.lower():
            raise AnnotationValidationError(f"{sample_id} 输出包含绝对路径或 pickle 内容")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(f"JSONL 解析失败 {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnnotationValidationError(f"JSONL 第 {line_number} 行不是对象: {path}")
            rows.append(value)
    return rows


def _quantize(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{field} 必须是数值") from exc
    if not math.isfinite(number) or min(abs(number - level) for level in ALLOWED_LEVELS) > 0.01:
        raise AnnotationValidationError(f"{field} 必须是 0/0.33/0.67/1")
    return min(ALLOWED_LEVELS, key=lambda level: abs(number - level))


def _validate_annotation(candidate: Mapping[str, object], annotation: Mapping[str, object]) -> dict[str, object]:
    candidate_id = str(candidate.get("id", ""))
    _reject_agent_metadata(annotation, sample_id=candidate_id or "<missing-id>")
    if str(annotation.get("id", "")) != candidate_id:
        raise AnnotationValidationError(f"标注 ID 与候选不一致: {candidate_id}")
    for field in ("textSha256", "previousTextSha256"):
        if str(annotation.get(field, "")) != str(candidate.get(field, "")):
            raise AnnotationValidationError(f"{candidate_id} 的 {field} 不匹配")
    status = str(annotation.get("status", ""))
    if status not in ALLOWED_STATUS:
        raise AnnotationValidationError(f"{candidate_id} 的 status 无效: {status}")
    confidence = str(annotation.get("confidence", ""))
    if confidence not in ALLOWED_CONFIDENCE:
        raise AnnotationValidationError(f"{candidate_id} 的 confidence 无效: {confidence}")
    primary = str(annotation.get("primaryEmotion", ""))
    if primary not in ALLOWED_PRIMARY:
        raise AnnotationValidationError(f"{candidate_id} 的 primaryEmotion 无效: {primary}")
    if not isinstance(annotation.get("needsReview"), bool):
        raise AnnotationValidationError(f"{candidate_id} 的 needsReview 必须是布尔值")
    hard_cases = annotation.get("hardCases", [])
    if not isinstance(hard_cases, list) or not all(isinstance(item, str) for item in hard_cases):
        raise AnnotationValidationError(f"{candidate_id} 的 hardCases 必须是字符串数组")
    emotions = annotation.get("emotions")
    if not isinstance(emotions, Mapping) or set(emotions) != set(EMOTION_NAMES):
        raise AnnotationValidationError(f"{candidate_id} 的 emotions 必须正好包含八个固定维度")
    values = {name: _quantize(emotions[name], f"{candidate_id}.emotions.{name}") for name in EMOTION_NAMES}
    intensity = _quantize(annotation.get("intensity"), f"{candidate_id}.intensity")
    is_base = all(value == 0.0 for value in values.values())
    if is_base != (primary == "base"):
        raise AnnotationValidationError(f"{candidate_id} 的 base/primaryEmotion 不一致")
    if is_base != (intensity == 0.0):
        raise AnnotationValidationError(f"{candidate_id} 的 base/intensity 不一致")
    if annotation.get("status") == "drop" and not annotation.get("needsReview"):
        # A dropped sample is allowed, but it should never silently enter the
        # accepted set.  Keeping this branch explicit makes the policy clear.
        pass
    result = dict(annotation)
    result["schema"] = ANNOTATION_SCHEMA
    result["emotions"] = values
    result["intensity"] = intensity
    return result


def validate_annotation_file(
    candidates_path: str | Path,
    annotation_path: str | Path,
    *,
    expected_input_sha256: str | None = None,
) -> dict[str, object]:
    candidates = list(iter_candidate_records(candidates_path))
    annotations = _read_jsonl(annotation_path)
    actual_input_sha = file_sha256(candidates_path)
    if expected_input_sha256 and actual_input_sha != expected_input_sha256:
        raise AnnotationValidationError("候选输入文件 SHA-256 不匹配")
    if len(candidates) != len(annotations):
        raise AnnotationValidationError(
            f"标注行数 {len(annotations)} 不等于候选行数 {len(candidates)}"
        )
    validated: list[dict[str, object]] = []
    seen: set[str] = set()
    for candidate, annotation in zip(candidates, annotations):
        candidate_id = str(candidate.get("id", ""))
        if candidate_id in seen:
            raise AnnotationValidationError(f"候选 ID 重复: {candidate_id}")
        seen.add(candidate_id)
        validated.append(_validate_annotation(candidate, annotation))
    return {
        "schema": ANNOTATION_SCHEMA,
        "candidateFileSha256": actual_input_sha,
        "candidateCount": len(candidates),
        "annotations": validated,
        "acceptedCount": sum(item["status"] == "accepted" for item in validated),
        "reviewCount": sum(item["status"] == "review" for item in validated),
        "dropCount": sum(item["status"] == "drop" for item in validated),
    }


def split_candidates(
    candidates_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 50,
    limit: int | None = None,
) -> dict[str, object]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    rows = list(iter_candidate_records(candidates_path))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        rows = rows[:limit]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        path = target / f"chunk-{start // chunk_size:04d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in chunk_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        chunks.append(
            {
                "chunkId": start // chunk_size,
                "path": str(path),
                "start": start,
                "end": start + len(chunk_rows),
                "rows": len(chunk_rows),
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema": ANNOTATION_SCHEMA,
        "input": str(Path(candidates_path).resolve()),
        "inputSha256": file_sha256(candidates_path),
        "rowCount": len(rows),
        "limited": limit is not None,
        "chunkSize": chunk_size,
        "chunks": chunks,
    }
    (target / "chunk-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _merge_value(left: float, right: float) -> float:
    average = (left + right) / 2.0
    return min(ALLOWED_LEVELS, key=lambda level: abs(average - level))


def merge_double_annotations(
    candidates_path: str | Path,
    annotation_a_path: str | Path,
    annotation_b_path: str | Path,
    output_dir: str | Path,
    *,
    license_status: str = "pending",
) -> dict[str, object]:
    if license_status not in {"pending", "test-only"}:
        raise ValueError("license_status 只能是 pending 或 test-only")
    first = validate_annotation_file(candidates_path, annotation_a_path)
    second = validate_annotation_file(
        candidates_path,
        annotation_b_path,
        expected_input_sha256=str(first["candidateFileSha256"]),
    )
    by_a = {str(item["id"]): item for item in first["annotations"]}
    by_b = {str(item["id"]): item for item in second["annotations"]}
    candidates = list(iter_candidate_records(candidates_path))
    accepted, review, audit = _merge_rows(
        candidates, first["annotations"], second["annotations"], license_status=license_status
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "accepted-pending-license.jsonl", accepted)
    _write_jsonl(target / "review-queue.jsonl", review)
    _write_jsonl(target / "audit.jsonl", audit)
    report = {
        "schema": ANNOTATION_SCHEMA,
        "candidateFileSha256": first["candidateFileSha256"],
        "candidateCount": len(candidates),
        "acceptedPendingLicense": len(accepted),
        "reviewCount": len(review),
        "auditFile": str(target / "audit.jsonl"),
        "licenseStatus": license_status,
    }
    (target / "merge-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _merge_rows(
    candidates: list[dict[str, object]],
    annotations_a: list[dict[str, object]],
    annotations_b: list[dict[str, object]],
    *,
    license_status: str = "pending",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Merge already validated rows while retaining an audit trail."""
    by_a = {str(item["id"]): item for item in annotations_a}
    by_b = {str(item["id"]): item for item in annotations_b}
    accepted: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        left, right = by_a[candidate_id], by_b[candidate_id]
        reasons: list[str] = []
        left_values = left["emotions"]
        right_values = right["emotions"]
        if any(abs(float(left_values[name]) - float(right_values[name])) > 0.33 for name in EMOTION_NAMES):
            reasons.append("emotion-difference")
        if abs(float(left["intensity"]) - float(right["intensity"])) > 0.33:
            reasons.append("intensity-difference")
        if left["primaryEmotion"] != right["primaryEmotion"]:
            reasons.append("primary-difference")
        if (left["primaryEmotion"] == "base") != (right["primaryEmotion"] == "base"):
            reasons.append("base-conflict")
        if (left["primaryEmotion"] == "calm") != (right["primaryEmotion"] == "calm") and (
            left["primaryEmotion"] == "base" or right["primaryEmotion"] == "base"
        ):
            reasons.append("calm-base-conflict")
        if any(
            item["status"] != "accepted"
            or item["needsReview"]
            or item["confidence"] == "low"
            or item["hardCases"]
            for item in (left, right)
        ):
            reasons.append("manual-review-flag")
        merged = dict(candidate)
        merged.update(
            {
                "emotions": {
                    name: _merge_value(float(left_values[name]), float(right_values[name]))
                    for name in EMOTION_NAMES
                },
                "intensity": _merge_value(float(left["intensity"]), float(right["intensity"])),
                "primaryEmotion": left["primaryEmotion"]
                if left["primaryEmotion"] == right["primaryEmotion"]
                else "mixed",
                "licenseStatus": license_status,
                "licenseEvidence": None,
            }
        )
        audit.append({"candidate": candidate, "annotationA": left, "annotationB": right, "reasons": reasons})
        if reasons:
            review.append({"candidate": candidate, "annotationA": left, "annotationB": right, "reasons": reasons})
        else:
            accepted.append(merged)
    return accepted, review, audit


def merge_annotation_tree(
    chunks_dir: str | Path,
    annotation_a_dir: str | Path,
    annotation_b_dir: str | Path,
    output_dir: str | Path,
    *,
    license_status: str = "pending",
) -> dict[str, object]:
    """Merge every completed pair in a chunk manifest.

    The operation is intentionally all-or-nothing: a missing pair or a bad
    hash aborts before writing aggregate files. This keeps a resumable batch
    from being mistaken for a complete training set.
    """
    root = Path(chunks_dir)
    if license_status not in {"pending", "test-only"}:
        raise ValueError("license_status 只能是 pending 或 test-only")
    manifest_path = root / "chunk-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationValidationError(f"无法读取 chunk-manifest.json: {manifest_path}") from exc
    if manifest.get("schema") != ANNOTATION_SCHEMA:
        raise AnnotationValidationError("chunk manifest schema 不匹配")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise AnnotationValidationError("chunk manifest 为空")
    a_root, b_root = Path(annotation_a_dir), Path(annotation_b_dir)
    missing: list[str] = []
    resolved: list[tuple[Path, Path, Path, dict[str, object]]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise AnnotationValidationError("chunk manifest 含非法条目")
        chunk_id = int(chunk.get("chunkId", -1))
        chunk_name = f"chunk-{chunk_id:04d}.jsonl"
        candidate_path = root / chunk_name
        a_path = a_root / chunk_name
        b_path = b_root / chunk_name
        if not candidate_path.is_file() or not a_path.is_file() or not b_path.is_file():
            missing.append(chunk_name)
            continue
        resolved.append((candidate_path, a_path, b_path, chunk))
    if missing:
        raise AnnotationValidationError(
            f"标注双份结果不完整，缺少 {len(missing)} 个 chunk: {', '.join(missing[:8])}"
        )

    accepted: list[dict[str, object]] = []
    review: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    chunk_reports: list[dict[str, object]] = []
    for candidate_path, a_path, b_path, chunk in resolved:
        expected_sha = str(chunk.get("sha256", ""))
        actual_sha = file_sha256(candidate_path)
        if expected_sha and actual_sha != expected_sha:
            raise AnnotationValidationError(f"{candidate_path.name} 的 chunk SHA-256 不匹配")
        first = validate_annotation_file(candidate_path, a_path, expected_input_sha256=actual_sha)
        second = validate_annotation_file(candidate_path, b_path, expected_input_sha256=actual_sha)
        part_accepted, part_review, part_audit = _merge_rows(
            list(iter_candidate_records(candidate_path)),
            first["annotations"],
            second["annotations"],
            license_status=license_status,
        )
        accepted.extend(part_accepted)
        review.extend(part_review)
        audit.extend(part_audit)
        chunk_reports.append(
            {
                "chunk": candidate_path.name,
                "candidateCount": first["candidateCount"],
                "acceptedPendingLicense": len(part_accepted),
                "reviewCount": len(part_review),
            }
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "accepted-pending-license.jsonl", accepted)
    _write_jsonl(target / "review-queue.jsonl", review)
    _write_jsonl(target / "audit.jsonl", audit)
    report = {
        "schema": ANNOTATION_SCHEMA,
        "candidateFileSha256": str(manifest.get("inputSha256", "")),
        "candidateCount": sum(int(item["candidateCount"]) for item in chunk_reports),
        "chunkCount": len(chunk_reports),
        "acceptedPendingLicense": len(accepted),
        "reviewCount": len(review),
        "auditFile": str(target / "audit.jsonl"),
        "licenseStatus": license_status,
        "chunks": chunk_reports,
    }
    (target / "merge-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def annotation_tree_status(
    chunks_dir: str | Path,
    annotation_a_dir: str | Path,
    annotation_b_dir: str | Path,
) -> dict[str, object]:
    """Return resumable progress without writing or trusting partial output."""
    root = Path(chunks_dir)
    manifest_path = root / "chunk-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationValidationError(f"无法读取 chunk-manifest.json: {manifest_path}") from exc
    chunks = manifest.get("chunks")
    if manifest.get("schema") != ANNOTATION_SCHEMA or not isinstance(chunks, list):
        raise AnnotationValidationError("chunk manifest 无效")
    a_root, b_root = Path(annotation_a_dir), Path(annotation_b_dir)
    complete = 0
    invalid: list[dict[str, str]] = []
    pending: list[str] = []
    a_counts = {"accepted": 0, "review": 0, "drop": 0}
    b_counts = {"accepted": 0, "review": 0, "drop": 0}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise AnnotationValidationError("chunk manifest 含非法条目")
        chunk_name = f"chunk-{int(chunk.get('chunkId', -1)):04d}.jsonl"
        candidate_path = root / chunk_name
        a_path, b_path = a_root / chunk_name, b_root / chunk_name
        if not a_path.is_file() or not b_path.is_file():
            pending.append(chunk_name)
            continue
        try:
            candidate_sha = file_sha256(candidate_path)
            expected_chunk_sha = str(chunk.get("sha256", ""))
            if expected_chunk_sha and candidate_sha != expected_chunk_sha:
                raise AnnotationValidationError(f"{chunk_name} 的 chunk SHA-256 不匹配")
            first = validate_annotation_file(candidate_path, a_path, expected_input_sha256=candidate_sha)
            second = validate_annotation_file(candidate_path, b_path, expected_input_sha256=candidate_sha)
        except (OSError, AnnotationValidationError) as exc:
            invalid.append({"chunk": chunk_name, "error": str(exc)})
            continue
        complete += 1
        for key in a_counts:
            a_counts[key] += int(first[f"{key}Count"])
            b_counts[key] += int(second[f"{key}Count"])
    return {
        "schema": ANNOTATION_SCHEMA,
        "candidateFileSha256": str(manifest.get("inputSha256", "")),
        "totalChunks": len(chunks),
        "completePairs": complete,
        "pendingPairs": len(pending),
        "invalidPairs": len(invalid),
        "pendingChunks": pending,
        "invalidChunks": invalid,
        "annotationA": a_counts,
        "annotationB": b_counts,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
