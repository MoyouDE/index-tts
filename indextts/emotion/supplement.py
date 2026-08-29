"""Integrate reviewed coverage labels into a prepared training snapshot.

The coverage backlog is intentionally annotated outside the original 20k
candidate pool.  This module joins only complete double-annotation output and
resolved review decisions, then assigns every supplemental row to the split
already owned by its work.  It never mutates the prepared base snapshot.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from .annotation import (
    AnnotationValidationError,
    _merge_rows,
    _read_jsonl,
    _validate_annotation,
    file_sha256,
    validate_annotation_file,
)
from .finalize import _as_example
from .schema import EMOTION_NAMES, load_jsonl_examples, validate_work_splits


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _load_reviewed_rows(
    merged_dir: Path,
    reviewed_dir: Path | None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    direct_path = merged_dir / "accepted-pending-license.jsonl"
    queue_path = merged_dir / "review-queue.jsonl"
    if not direct_path.is_file() or not queue_path.is_file():
        raise AnnotationValidationError("覆盖合并目录缺少 accepted 或 review queue")
    direct = _read_jsonl(direct_path)
    queue = _read_jsonl(queue_path)
    reviewed: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    if queue:
        if reviewed_dir is None:
            raise AnnotationValidationError(f"仍有 {len(queue)} 条 review queue 未复核")
        report_path = reviewed_dir / "review-report.json"
        accepted_path = reviewed_dir / "reviewed-accepted.jsonl"
        dropped_path = reviewed_dir / "reviewed-drop.jsonl"
        audit_path = reviewed_dir / "review-audit.jsonl"
        decisions_path = merged_dir / "review-decisions.jsonl"
        if not all(
            path.is_file()
            for path in (report_path, accepted_path, dropped_path, audit_path, decisions_path)
        ):
            raise AnnotationValidationError("复核目录不完整")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if str(report.get("queueSha256", "")) != file_sha256(queue_path):
            raise AnnotationValidationError("复核报告与当前 review queue 哈希不一致")
        if int(report.get("queueCount", -1)) != len(queue):
            raise AnnotationValidationError("复核报告行数与 review queue 不一致")
        reviewed = _read_jsonl(accepted_path)
        dropped = _read_jsonl(dropped_path)
        if len(reviewed) + len(dropped) != len(queue):
            raise AnnotationValidationError("复核 accepted/drop 未覆盖完整 review queue")
        if int(report.get("reviewedAccepted", -1)) != len(reviewed):
            raise AnnotationValidationError("复核 accepted 行数与报告不一致")
        if int(report.get("reviewedDrop", -1)) != len(dropped):
            raise AnnotationValidationError("复核 drop 行数与报告不一致")
        for report_key, path in (
            ("decisionsSha256", decisions_path),
            ("reviewedAcceptedSha256", accepted_path),
            ("reviewedDropSha256", dropped_path),
            ("reviewAuditSha256", audit_path),
        ):
            if str(report.get(report_key, "")) != file_sha256(path):
                raise AnnotationValidationError(f"复核报告 {report_key} 哈希不一致")

        decisions = _read_jsonl(decisions_path)
        review_audit = _read_jsonl(audit_path)
        if len(decisions) != len(queue) or len(review_audit) != len(queue):
            raise AnnotationValidationError("复核 decisions/audit 未覆盖完整 review queue")
        output_by_id: dict[str, tuple[str, dict[str, object]]] = {}
        for status, rows in (("accepted", reviewed), ("drop", dropped)):
            for row in rows:
                candidate_id = str(row.get("id", ""))
                if not candidate_id or candidate_id in output_by_id:
                    raise AnnotationValidationError(f"复核输出存在空或重复 id: {candidate_id!r}")
                output_by_id[candidate_id] = (status, row)
        for item, decision, audit in zip(queue, decisions, review_audit):
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                raise AnnotationValidationError("review queue 缺少 candidate")
            candidate_id = str(candidate.get("id", ""))
            validated = _validate_annotation(candidate, decision)
            if audit.get("candidate") != candidate:
                raise AnnotationValidationError(f"{candidate_id} review audit candidate 不一致")
            if audit.get("annotationA") != item.get("annotationA"):
                raise AnnotationValidationError(f"{candidate_id} review audit 缺少或篡改 annotationA")
            if audit.get("annotationB") != item.get("annotationB"):
                raise AnnotationValidationError(f"{candidate_id} review audit 缺少或篡改 annotationB")
            if audit.get("originalReasons", []) != item.get("reasons", []):
                raise AnnotationValidationError(f"{candidate_id} review audit 冲突原因不一致")
            if audit.get("decision") != validated:
                raise AnnotationValidationError(f"{candidate_id} review audit 最终决断不一致")
            output = output_by_id.get(candidate_id)
            if output is None or output[0] != validated["status"]:
                raise AnnotationValidationError(f"{candidate_id} 最终决断未写入对应 accepted/drop")
            final = output[1]
            for field in (
                "id",
                "workId",
                "previousText",
                "text",
                "sentenceType",
                "textSha256",
                "previousTextSha256",
            ):
                if final.get(field) != candidate.get(field):
                    raise AnnotationValidationError(f"{candidate_id} 复核输出 {field} 与候选不一致")
            for field in ("emotions", "intensity", "primaryEmotion"):
                if final.get(field) != validated.get(field):
                    raise AnnotationValidationError(f"{candidate_id} 复核输出 {field} 与最终决断不一致")
    elif reviewed_dir is not None and (reviewed_dir / "review-report.json").exists():
        raise AnnotationValidationError("review queue 为空但提供了复核结果")
    return direct + reviewed, {
        "directAccepted": len(direct),
        "reviewQueue": len(queue),
        "reviewedAccepted": len(reviewed),
        "reviewedDrop": len(dropped),
    }


def _load_reviewed_tree(
    merged_root: Path,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, object]]:
    """Load only fully merged and fully adjudicated chunk directories.

    Coverage annotation is intentionally allowed to stop after the selected
    chunks close the quality gaps.  Unlike ``merge_annotation_tree`` this
    reader therefore does not require every backlog chunk, but it does require
    every chunk that is present to have a complete merge report, original A/B
    audit trail, and (when needed) a hash-bound review result.
    """

    chunks = sorted(
        (path for path in merged_root.glob("chunk-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not chunks:
        raise AnnotationValidationError(f"没有找到覆盖合并分片: {merged_root}")
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    source_hashes: dict[str, object] = {}
    seen_ids: set[str] = set()
    candidates_root = merged_root.parent / "chunks"
    annotation_a_root = merged_root.parent / "annotations-a"
    annotation_b_root = merged_root.parent / "annotations-b"
    for chunk in chunks:
        report_path = chunk / "merge-report.json"
        audit_path = chunk / "audit.jsonl"
        if not report_path.is_file() or not audit_path.is_file():
            raise AnnotationValidationError(f"分片 {chunk.name} 缺少 merge report 或 A/B audit")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnotationValidationError(f"分片 {chunk.name} merge report 无法读取") from exc
        direct_path = chunk / "accepted-pending-license.jsonl"
        queue_path = chunk / "review-queue.jsonl"
        if not direct_path.is_file() or not queue_path.is_file():
            raise AnnotationValidationError(f"分片 {chunk.name} 缺少 accepted 或 review queue")
        direct_rows = _read_jsonl(direct_path)
        queue_rows = _read_jsonl(queue_path)
        audit_rows = _read_jsonl(audit_path)
        direct_count = len(direct_rows)
        queue_count = len(queue_rows)
        candidate_count = int(report.get("candidateCount", -1))
        if direct_count != int(report.get("acceptedPendingLicense", -1)):
            raise AnnotationValidationError(f"分片 {chunk.name} 直接接收行数与 merge report 不一致")
        if queue_count != int(report.get("reviewCount", -1)):
            raise AnnotationValidationError(f"分片 {chunk.name} 复核行数与 merge report 不一致")
        if direct_count + queue_count != candidate_count:
            raise AnnotationValidationError(f"分片 {chunk.name} merge 行数未覆盖全部候选")
        if len(audit_rows) != candidate_count:
            raise AnnotationValidationError(f"分片 {chunk.name} A/B audit 行数不完整")

        candidate_path = candidates_root / f"{chunk.name}.jsonl"
        if not candidate_path.is_file():
            raise AnnotationValidationError(f"分片 {chunk.name} 缺少原始盲标候选")
        candidate_sha = file_sha256(candidate_path)
        if candidate_sha != str(report.get("candidateFileSha256", "")):
            raise AnnotationValidationError(f"分片 {chunk.name} 候选哈希与 merge report 不一致")

        annotation_a_path = annotation_a_root / f"{chunk.name}.jsonl"
        annotation_b_path = annotation_b_root / f"{chunk.name}.jsonl"
        if not annotation_a_path.is_file() or not annotation_b_path.is_file():
            raise AnnotationValidationError(f"分片 {chunk.name} 缺少原始 A/B 标注")
        first = validate_annotation_file(candidate_path, annotation_a_path)
        second = validate_annotation_file(
            candidate_path,
            annotation_b_path,
            expected_input_sha256=candidate_sha,
        )
        license_status = str(report.get("licenseStatus", ""))
        if license_status not in {"pending", "test-only"}:
            raise AnnotationValidationError(f"分片 {chunk.name} licenseStatus 无效")
        expected_direct, expected_queue, expected_audit = _merge_rows(
            _read_jsonl(candidate_path),
            first["annotations"],
            second["annotations"],
            license_status=license_status,
        )
        if direct_rows != expected_direct:
            raise AnnotationValidationError(f"分片 {chunk.name} accepted 与原始 A/B 重算结果不一致")
        if queue_rows != expected_queue:
            raise AnnotationValidationError(f"分片 {chunk.name} review queue 与原始 A/B 重算结果不一致")
        if audit_rows != expected_audit:
            raise AnnotationValidationError(f"分片 {chunk.name} A/B audit 与原始标注不一致")

        reviewed_dir = chunk / "reviewed" if queue_count else None
        chunk_rows, chunk_counts = _load_reviewed_rows(chunk, reviewed_dir)
        for row in chunk_rows:
            candidate_id = str(row.get("id", ""))
            if not candidate_id or candidate_id in seen_ids:
                raise AnnotationValidationError(f"覆盖合并树存在空或重复 id: {candidate_id!r}")
            seen_ids.add(candidate_id)
            rows.append(row)
        counts.update(chunk_counts)
        source_hashes[chunk.name] = {
            "candidate": candidate_sha,
            "annotationA": file_sha256(annotation_a_path),
            "annotationB": file_sha256(annotation_b_path),
            "mergeReport": file_sha256(report_path),
            "doubleAnnotationAudit": file_sha256(audit_path),
            "directAccepted": file_sha256(direct_path),
            "reviewQueue": file_sha256(queue_path),
            "reviewReport": (
                file_sha256(reviewed_dir / "review-report.json")
                if reviewed_dir is not None
                else None
            ),
            "reviewDecisions": (
                file_sha256(chunk / "review-decisions.jsonl")
                if reviewed_dir is not None
                else None
            ),
            "reviewedAccepted": (
                file_sha256(reviewed_dir / "reviewed-accepted.jsonl")
                if reviewed_dir is not None
                else None
            ),
            "reviewedDrop": (
                file_sha256(reviewed_dir / "reviewed-drop.jsonl")
                if reviewed_dir is not None
                else None
            ),
            "reviewAudit": (
                file_sha256(reviewed_dir / "review-audit.jsonl")
                if reviewed_dir is not None
                else None
            ),
        }
    counts["chunks"] = len(chunks)
    return rows, dict(counts), source_hashes


def _positive_counts(rows: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        emotions = row.get("emotions")
        if not isinstance(emotions, Mapping):
            continue
        for name in EMOTION_NAMES:
            if float(emotions.get(name, 0.0)) > 0.0:
                counts[name] += 1
    return counts


def integrate_reviewed_supplement(
    base_dir: str | Path,
    merged_dir: str | Path,
    output_dir: str | Path,
    *,
    reviewed_dir: str | Path | None = None,
    brighter_dir: str | Path | None = None,
    minimum_positive: int = 1_000,
    license_id: str = "PROPRIETARY-AUTHORIZED",
) -> dict[str, object]:
    """Create a new test-training snapshot with a reviewed coverage supplement."""

    if minimum_positive < 0:
        raise ValueError("minimum_positive 不能为负数")
    base_root = Path(base_dir)
    merge_root = Path(merged_dir)
    review_root = Path(reviewed_dir) if reviewed_dir is not None else None
    if (merge_root / "accepted-pending-license.jsonl").is_file():
        supplement, review_counts = _load_reviewed_rows(merge_root, review_root)
        merge_hashes: dict[str, object] = {
            "mergedAccepted": file_sha256(merge_root / "accepted-pending-license.jsonl"),
            "mergedReviewQueue": file_sha256(merge_root / "review-queue.jsonl"),
        }
    else:
        if review_root is not None:
            raise AnnotationValidationError("分片合并树会自动读取各 chunk/reviewed，不接受独立 reviewed_dir")
        supplement, review_counts, merge_hashes = _load_reviewed_tree(merge_root)
    if not supplement:
        raise AnnotationValidationError("覆盖补充集没有可接收样本")

    split_rows: dict[str, list[dict[str, object]]] = {}
    split_examples = {}
    work_owner: dict[str, str] = {}
    seen_ids: set[str] = set()
    context_owner: dict[tuple[str, str, str, str], str] = {}
    for split in ("train", "dev", "test"):
        path = base_root / f"{split}.jsonl"
        rows = _read_jsonl(path)
        examples = load_jsonl_examples(path)
        split_rows[split] = rows
        split_examples[split] = examples
        for row, example in zip(rows, examples):
            previous_owner = work_owner.setdefault(example.work_id, split)
            if previous_owner != split:
                raise AnnotationValidationError(f"基础快照作品 {example.work_id} 跨 split")
            if example.example_id in seen_ids:
                raise AnnotationValidationError(f"基础快照重复 id: {example.example_id}")
            seen_ids.add(example.example_id)
            context_owner.setdefault(
                (example.work_id, example.previous_text, example.text, example.sentence_type),
                example.example_id,
            )
    validate_work_splits(split_examples)

    supplemental_by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    deduplicated: list[dict[str, object]] = []
    for row in supplement:
        candidate_id = str(row.get("id", ""))
        work_id = str(row.get("workId", ""))
        if candidate_id in seen_ids:
            raise AnnotationValidationError(f"覆盖补充重复 id: {candidate_id}")
        split = work_owner.get(work_id)
        if split is None:
            raise AnnotationValidationError(f"覆盖补充作品 {work_id!r} 不在基础快照中")
        context = (
            work_id,
            str(row.get("previousText", "")),
            str(row.get("text", "")),
            str(row.get("sentenceType", "")),
        )
        existing_id = context_owner.get(context)
        if existing_id is not None:
            deduplicated.append(
                {
                    "droppedId": candidate_id,
                    "keptId": existing_id,
                    "reason": "exact-context-duplicate",
                    "textSha256": str(row.get("textSha256", "")),
                    "workId": work_id,
                }
            )
            continue
        seen_ids.add(candidate_id)
        context_owner[context] = candidate_id
        annotation_status = (
            "coverage-human-reviewed"
            if str(row.get("reviewStatus", "")) == "human-reviewed"
            else "coverage-double-accepted"
        )
        supplemental_by_split[split].append(
            _as_example(row, license_id=license_id, annotation_status=annotation_status)
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    dedup_path = target / "supplement-dedup-audit.jsonl"
    _write_jsonl(dedup_path, deduplicated)
    output_counts: dict[str, int] = {}
    output_hashes: dict[str, str] = {}
    combined_rows: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "dev", "test"):
        extra = sorted(supplemental_by_split[split], key=lambda row: str(row["id"]))
        combined = [*split_rows[split], *extra]
        combined_rows[split] = combined
        path = target / f"{split}.jsonl"
        output_counts[split] = _write_jsonl(path, combined)
        output_hashes[split] = file_sha256(path)

    parsed = {
        split: load_jsonl_examples(target / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    validate_work_splits(parsed)

    training_positive = _positive_counts(combined_rows["train"])
    brighter_count = 0
    if brighter_dir is not None:
        brighter_path = Path(brighter_dir) / "brighter-chn-train.jsonl"
        if not brighter_path.is_file():
            raise AnnotationValidationError(f"缺少本地 BRIGHTER 训练数据: {brighter_path}")
        brighter_rows = _read_jsonl(brighter_path)
        brighter_count = len(brighter_rows)
        training_positive.update(_positive_counts(brighter_rows))
    gaps = {
        name: max(0, minimum_positive - training_positive[name])
        for name in EMOTION_NAMES
    }
    report: dict[str, object] = {
        "schema": "readest-emotion-training-supplement-v1",
        "trainingStarted": False,
        "testTrainingReady": not any(gaps.values()),
        "formalTrainingReady": False,
        "baseSplitCounts": {split: len(split_rows[split]) for split in split_rows},
        "supplementSplitCounts": {
            split: len(supplemental_by_split[split]) for split in supplemental_by_split
        },
        "outputSplitCounts": output_counts,
        "review": review_counts,
        "supplementDeduplicated": len(deduplicated),
        "brighterTrainCount": brighter_count,
        "minimumPositivePerEmotion": minimum_positive,
        "trainingPositiveExamples": {
            name: training_positive[name] for name in EMOTION_NAMES
        },
        "remainingPositiveGaps": gaps,
        "sourceHashes": {
            "base": {
                split: file_sha256(base_root / f"{split}.jsonl")
                for split in ("train", "dev", "test")
            },
            "coverage": merge_hashes,
            "dedupAudit": file_sha256(dedup_path),
        },
        "outputs": output_hashes,
    }
    (target / "supplement-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
