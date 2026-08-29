"""Materialize a reviewed, test-only EmotionExample dataset.

The per-chunk merge tree intentionally keeps direct agreements and review
decisions separate.  This module is the final, reproducible step that joins
those records, enforces work-level splits, and writes the schema consumed by
the MacBERT trainer.  It never changes source candidates or annotation files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .annotation import AnnotationValidationError, _read_jsonl, file_sha256
from .schema import EMOTION_NAMES, load_jsonl_examples, validate_work_splits


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _as_example(row: dict[str, object], *, license_id: str, annotation_status: str) -> dict[str, object]:
    emotions = row.get("emotions")
    if not isinstance(emotions, dict) or set(emotions) != set(EMOTION_NAMES):
        raise AnnotationValidationError(f"{row.get('id', '<missing-id>')} 情感维度不完整")
    return {
        "schemaVersion": 1,
        "id": str(row["id"]),
        "workId": str(row["workId"]),
        "previousText": str(row.get("previousText", "")),
        "text": str(row["text"]),
        "sentenceType": str(row["sentenceType"]),
        "emotions": {name: float(emotions[name]) for name in EMOTION_NAMES},
        "labelMask": {name: 1.0 for name in EMOTION_NAMES},
        "intensity": float(row["intensity"]),
        # The source corpus is explicitly test-only in this materialization.
        # The placeholder license id keeps the trainer schema valid while the
        # manifest prevents this dataset from being treated as a release.
        "licenseId": license_id,
        "source": str(row.get("source", "speaker-id")),
        "annotationStatus": annotation_status,
        "licenseStatus": "test-only",
    }


def _split_work_ids(work_ids: list[str]) -> dict[str, str]:
    """Assign works deterministically to train/dev/test without leakage."""
    assignment: dict[str, str] = {}
    for index, work_id in enumerate(sorted(work_ids)):
        bucket = index % 10
        assignment[work_id] = "train" if bucket < 7 else ("dev" if bucket == 7 else "test")
    return assignment


def finalize_reviewed_dataset(
    merged_root: str | Path,
    output_dir: str | Path,
    *,
    candidates_path: str | Path | None = None,
    license_id: str = "PROPRIETARY-AUTHORIZED",
) -> dict[str, object]:
    """Join all reviewed chunk outputs and write test-only train/dev/test JSONL."""

    if license_id not in {"CC-BY-4.0", "CC0-1.0", "PUBLIC-DOMAIN", "PROPRIETARY-AUTHORIZED"}:
        raise ValueError("license_id 不在数据 schema 白名单中")
    root = Path(merged_root)
    chunks = sorted(root.glob("chunk-*"), key=lambda path: path.name)
    if not chunks:
        raise AnnotationValidationError(f"没有找到合并分片: {root}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    direct_count = 0
    reviewed_count = 0
    queue_count = 0
    reviewed_drop_count = 0
    for chunk in chunks:
        direct_path = chunk / "accepted-pending-license.jsonl"
        if direct_path.is_file():
            for row in _read_jsonl(direct_path):
                candidate_id = str(row.get("id", ""))
                if candidate_id in seen:
                    raise AnnotationValidationError(f"最终数据重复 id: {candidate_id}")
                seen.add(candidate_id)
                rows.append(_as_example(row, license_id=license_id, annotation_status="double-accepted"))
                direct_count += 1

        queue_path = chunk / "review-queue.jsonl"
        if not queue_path.is_file():
            continue
        queue = _read_jsonl(queue_path)
        queue_count += len(queue)
        reviewed_dir = chunk / "reviewed"
        report_path = reviewed_dir / "review-report.json"
        if not report_path.is_file():
            raise AnnotationValidationError(f"分片 {chunk.name} 缺少复核报告")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("queueCount", -1)) != len(queue):
            raise AnnotationValidationError(f"分片 {chunk.name} 复核行数不一致")
        if str(report.get("queueSha256", "")) != file_sha256(queue_path):
            raise AnnotationValidationError(f"分片 {chunk.name} 复核队列哈希不一致")
        accepted_path = reviewed_dir / "reviewed-accepted.jsonl"
        drop_path = reviewed_dir / "reviewed-drop.jsonl"
        accepted = _read_jsonl(accepted_path) if accepted_path.is_file() else []
        dropped = _read_jsonl(drop_path) if drop_path.is_file() else []
        if len(accepted) + len(dropped) != len(queue):
            raise AnnotationValidationError(f"分片 {chunk.name} 复核 accepted/drop 未覆盖完整队列")
        reviewed_drop_count += len(dropped)
        for row in accepted:
            candidate_id = str(row.get("id", ""))
            if candidate_id in seen:
                raise AnnotationValidationError(f"最终数据重复 id: {candidate_id}")
            seen.add(candidate_id)
            rows.append(_as_example(row, license_id=license_id, annotation_status="human-reviewed"))
            reviewed_count += 1

    work_assignment = _split_work_ids(sorted({str(row["workId"]) for row in rows}))
    splits: dict[str, list[dict[str, object]]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        splits[work_assignment[str(row["workId"])]].append(row)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in splits.items():
        _write_jsonl(target / f"{split_name}.jsonl", split_rows)
    parsed = {
        name: load_jsonl_examples(target / f"{name}.jsonl") for name in splits
    }
    validate_work_splits(parsed)

    positive = Counter()
    sentence_types = Counter()
    for row in rows:
        sentence_types[str(row["sentenceType"])] += 1
        for name in EMOTION_NAMES:
            if float(row["emotions"][name]) > 0.0:
                positive[name] += 1
    candidate_count = len(_read_jsonl(candidates_path)) if candidates_path else None
    report: dict[str, object] = {
        "schemaVersion": 1,
        "datasetStatus": "candidate-unvalidated",
        "licenseStatus": "test-only",
        "licenseId": license_id,
        "candidateCount": candidate_count,
        "acceptedCount": len(rows),
        "doubleAcceptedCount": direct_count,
        "reviewedAcceptedCount": reviewed_count,
        "reviewQueueRows": queue_count,
        "reviewedQueueRows": queue_count,
        "reviewedDropCount": reviewed_drop_count,
        "workCount": len(work_assignment),
        "works": {name: sorted(work for work, split in work_assignment.items() if split == name) for name in splits},
        "splitCounts": {name: len(items) for name, items in splits.items()},
        "splitWorks": {name: len({str(item["workId"]) for item in items}) for name, items in splits.items()},
        "positiveExamples": dict(positive),
        "sentenceTypes": dict(sentence_types),
        "sourceCandidateSha256": file_sha256(candidates_path) if candidates_path else None,
        "outputs": {name: file_sha256(target / f"{name}.jsonl") for name in splits},
    }
    (target / "dataset-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
