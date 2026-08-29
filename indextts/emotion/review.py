"""Strict resolution of double-annotation review queues.

Reviewers receive only the merged queue for a chunk and return one final
annotation-shaped record per queued candidate.  The resolver deliberately
keeps the original A/B records untouched and writes a separate audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

from .annotation import (
    ANNOTATION_SCHEMA,
    AnnotationValidationError,
    _read_jsonl,
    _validate_annotation,
    file_sha256,
)


REVIEW_SCHEMA = "readest-emotion-review-v1"


def resolve_review_queue(
    queue_path: str | Path,
    decisions_path: str | Path,
    output_dir: str | Path,
    *,
    license_status: str = "pending",
) -> dict[str, object]:
    """Validate and materialize a complete set of human-review decisions.

    ``decisions_path`` must contain exactly one annotation-shaped row for each
    queue row, in the same order.  Decisions may only be ``accepted`` or
    ``drop`` and must set ``needsReview`` to false; an unresolved ``review``
    value is rejected instead of silently entering a training set.
    """

    if license_status not in {"pending", "test-only"}:
        raise ValueError("license_status 只能是 pending 或 test-only")
    queue = _read_jsonl(queue_path)
    decisions = _read_jsonl(decisions_path)
    if len(queue) != len(decisions):
        raise AnnotationValidationError(
            f"复核行数 {len(decisions)} 不等于 review queue 行数 {len(queue)}"
        )
    queue_sha = file_sha256(queue_path)
    accepted: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, (item, decision) in enumerate(zip(queue, decisions), 1):
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            raise AnnotationValidationError(f"review queue 第 {index} 行缺少 candidate")
        candidate_id = str(candidate.get("id", ""))
        if candidate_id in seen:
            raise AnnotationValidationError(f"review queue ID 重复: {candidate_id}")
        seen.add(candidate_id)
        if str(decision.get("id", "")) != candidate_id:
            raise AnnotationValidationError(f"复核 ID 与候选不一致: {candidate_id}")
        if str(decision.get("textSha256", "")) != str(candidate.get("textSha256", "")):
            raise AnnotationValidationError(f"{candidate_id} 的 textSha256 不匹配")
        if str(decision.get("previousTextSha256", "")) != str(candidate.get("previousTextSha256", "")):
            raise AnnotationValidationError(f"{candidate_id} 的 previousTextSha256 不匹配")
        if decision.get("status") not in {"accepted", "drop"}:
            raise AnnotationValidationError(f"{candidate_id} 复核后 status 必须是 accepted 或 drop")
        if decision.get("needsReview") is not False:
            raise AnnotationValidationError(f"{candidate_id} 复核后 needsReview 必须为 false")
        validated = _validate_annotation(candidate, decision)
        final = dict(candidate)
        final.update(
            {
                "schema": ANNOTATION_SCHEMA,
                "emotions": validated["emotions"],
                "intensity": validated["intensity"],
                "primaryEmotion": validated["primaryEmotion"],
                "licenseStatus": license_status,
                "licenseEvidence": None,
                "reviewStatus": "human-reviewed",
                "reviewSchema": REVIEW_SCHEMA,
            }
        )
        audit.append(
            {
                "candidate": candidate,
                "annotationA": item.get("annotationA"),
                "annotationB": item.get("annotationB"),
                "originalReasons": item.get("reasons", []),
                "decision": validated,
                "queueSha256": queue_sha,
            }
        )
        (accepted if validated["status"] == "accepted" else dropped).append(final)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    accepted_path = target / "reviewed-accepted.jsonl"
    dropped_path = target / "reviewed-drop.jsonl"
    audit_path = target / "review-audit.jsonl"
    _write_jsonl(accepted_path, accepted)
    _write_jsonl(dropped_path, dropped)
    _write_jsonl(audit_path, audit)
    report = {
        "schema": REVIEW_SCHEMA,
        "queueSha256": queue_sha,
        "queueCount": len(queue),
        "reviewedAccepted": len(accepted),
        "reviewedDrop": len(dropped),
        "licenseStatus": license_status,
        "decisionsSha256": file_sha256(decisions_path),
        "reviewedAcceptedSha256": file_sha256(accepted_path),
        "reviewedDropSha256": file_sha256(dropped_path),
        "reviewAuditSha256": file_sha256(audit_path),
    }
    (target / "review-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
