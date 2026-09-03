"""Continuous eight-dimensional emotion pseudo-label data v3 tooling.

This pipeline intentionally keeps the raw Qwen target-text-only output and
uses agent review only for directional conflicts.  It never
applies IndexTTS runtime emotion post-processing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .annotation import file_sha256
from .qwen_continuous import strict_parse_raw_emotions
from .schema import EMOTION_NAMES


CANDIDATE_SCHEMA = "readest-emotion-continuous-candidate-v3"
QWEN_SCHEMA = "readest-emotion-qwen-raw-v3"
REVIEW_QUEUE_SCHEMA = "readest-emotion-continuous-review-queue-v3"
REVIEW_SCHEMA = "readest-emotion-continuous-review-v3"
QWEN_EXCLUSION_SCHEMA = "readest-emotion-qwen-exclusion-v3"
FINAL_SCHEMA = "readest-emotion-continuous-v3"
SCHEMA_VERSION = 3
SIGNIFICANT_THRESHOLD = 0.10
DOMINANCE_TOLERANCE = 0.10
BRIGHTER_OBSERVED_DIRECTIONS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "surprised",
)
DEFAULT_SPLIT_COUNTS = {"train": 16164, "dev": 3097, "test": 6777}
_SPLITS = ("train", "dev", "test")
_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ("『", "』"), ("‘", "’"), ('"', '"'), ("'", "'"))


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path, *, allow_empty: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败 {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象: {path}")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"JSONL 文件为空: {path}")
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _as_emotions(value: object, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(EMOTION_NAMES):
        raise ValueError(f"{field} 必须正好包含八个固定情感维度")
    result: dict[str, float] = {}
    for name in EMOTION_NAMES:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field}.{name} 必须是数值")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{field}.{name} 必须是 [0,1] 有限数值")
        result[name] = number
    return result


def _ordered_directions(values: Mapping[str, float]) -> list[str]:
    return [name for name in EMOTION_NAMES if float(values[name]) > 0.0]


def infer_brighter_sentence_type(text: str) -> str:
    """Conservatively classify only a fully quote-wrapped target as dialogue."""
    target = str(text).strip()
    if len(target) >= 2 and any(target.startswith(left) and target.endswith(right) for left, right in _QUOTE_PAIRS):
        return "dialogue"
    return "narration"


def _infer_split(path: Path) -> str:
    lowered = path.stem.lower()
    matches = [split for split in _SPLITS if split in lowered]
    if len(matches) != 1:
        raise ValueError(f"无法从文件名推断 split: {path}")
    return matches[0]


def _split_paths(value: object) -> list[tuple[str, Path]]:
    if isinstance(value, Mapping):
        result = [(str(split), Path(path)) for split, path in value.items()]
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
            result = [(_infer_split(item), item) for item in files]
        else:
            result = [(_infer_split(path), path)]
    elif isinstance(value, Sequence):
        result = [(_infer_split(Path(path)), Path(path)) for path in value]
    else:
        raise TypeError("split 路径必须是映射、路径或路径序列")
    seen: set[str] = set()
    normalized: list[tuple[str, Path]] = []
    for split, path in result:
        if split not in _SPLITS or split in seen:
            raise ValueError(f"split 无效或重复: {split}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(split)
        normalized.append((split, path))
    return sorted(normalized, key=lambda item: _SPLITS.index(item[0]))


def _audit_by_id(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path, allow_empty=True):
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"旧审计 ID 缺失或重复: {sample_id or '<empty>'}")
        result[sample_id] = row
    return result


def _restored_audit_evidence(
    row: Mapping[str, object],
    *,
    sample_id: str,
    work_id: str,
    audit_row: Mapping[str, object],
) -> tuple[dict[str, float], str]:
    """Validate historical adjudication before recovering its original directions."""
    if audit_row.get("schema") != "readest-emotion-training-prep-v1":
        raise ValueError(f"旧恢复审计 schema 无效: {sample_id}")
    if audit_row.get("id") != sample_id:
        raise ValueError(f"旧恢复审计 ID 不匹配: {sample_id}")
    historical_split = audit_row.get("split")
    if historical_split not in _SPLITS:
        raise ValueError(f"旧恢复审计 split 无效: {sample_id}")
    if audit_row.get("workId") != work_id:
        raise ValueError(f"旧恢复审计 workId 不匹配: {sample_id}")
    chosen_source = str(row.get("adjudicationChosenSource", ""))
    if audit_row.get("chosenSource") != chosen_source:
        raise ValueError(f"旧恢复审计 chosenSource 不匹配: {sample_id}")
    current_emotions = _as_emotions(row.get("emotions"), f"{sample_id}.currentEmotions")
    final_emotions = _as_emotions(audit_row.get("finalEmotions"), f"{sample_id}.auditFinalEmotions")
    if final_emotions != current_emotions:
        raise ValueError(f"旧恢复审计 finalEmotions 与 v2 标签不匹配: {sample_id}")
    original_emotions = _as_emotions(
        audit_row.get("originalEmotions"), f"{sample_id}.auditOriginalEmotions"
    )
    return original_emotions, str(historical_split)


def _candidate_from_legacy(
    row: Mapping[str, object], split: str, source_kind: str, audit: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, object], bool, str | None]:
    sample_id = str(row.get("id", "")).strip()
    work_id = str(row.get("workId", "")).strip()
    text = str(row.get("text", "")).strip()
    if not sample_id or not work_id or not text:
        raise ValueError(f"旧数据缺少 id/workId/text: {sample_id or '<empty>'}")
    if source_kind == "novel":
        observed = list(EMOTION_NAMES)
        evidence_value = row.get("emotions")
        restored = str(row.get("adjudicationChosenSource", "")) in {"qwen", "revised"}
        if restored:
            audit_row = audit.get(sample_id)
            if audit_row is None:
                raise ValueError(f"旧 Qwen/revised 记录缺少 originalEmotions 审计: {sample_id}")
            evidence_value, historical_audit_split = _restored_audit_evidence(
                row,
                sample_id=sample_id,
                work_id=work_id,
                audit_row=audit_row,
            )
        else:
            historical_audit_split = None
        sentence_type = str(row.get("sentenceType", ""))
        if sentence_type not in {"dialogue", "narration"}:
            raise ValueError(f"小说 sentenceType 无效: {sample_id}")
    else:
        restored = False
        historical_audit_split = None
        mask = row.get("labelMask")
        if not isinstance(mask, Mapping) or set(mask) != set(EMOTION_NAMES):
            raise ValueError(f"BRIGHTER labelMask 不完整: {sample_id}")
        observed = []
        for name in EMOTION_NAMES:
            flag = mask[name]
            if isinstance(flag, bool) or not isinstance(flag, (int, float)) or float(flag) not in {0.0, 1.0}:
                raise ValueError(f"BRIGHTER labelMask 非 0/1: {sample_id}.{name}")
            if float(flag) == 1.0:
                observed.append(name)
        if set(observed) != set(BRIGHTER_OBSERVED_DIRECTIONS):
            raise ValueError(f"BRIGHTER 必须只观察既定六维: {sample_id}")
        evidence_value = row.get("emotions")
        sentence_type = infer_brighter_sentence_type(text)
    evidence = _as_emotions(evidence_value, f"{sample_id}.evidence")
    old_directions = [name for name in EMOTION_NAMES if name in observed and evidence[name] > 0.0]
    candidate: dict[str, object] = {
        "schema": CANDIDATE_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "id": sample_id,
        "split": split,
        "workId": work_id,
        "previousText": str(row.get("previousText", "")).strip(),
        "text": text,
        "sentenceType": sentence_type,
        "sourceKind": source_kind,
        "source": str(row.get("source", "")).strip(),
        "licenseId": str(row.get("licenseId", "")).strip(),
        "licenseStatus": str(row.get("licenseStatus", "approved")).strip(),
        "observedDirections": [name for name in EMOTION_NAMES if name in observed],
        "oldDirections": old_directions,
        "oldDominantDirections": list(old_directions),
        "textSha256": _text_hash(text),
    }
    if not candidate["source"] or not candidate["licenseId"]:
        raise ValueError(f"旧数据缺少 source/licenseId: {sample_id}")
    candidate["inputSha256"] = _canonical_hash(candidate)
    return candidate, restored, historical_audit_split


def prepare_continuous_v3(novel_paths, brighter_paths, adjudication_audit_path, output_dir):
    """Build deterministic v3 candidates without copying legacy strengths."""
    audit = _audit_by_id(adjudication_audit_path)
    rows: list[dict[str, object]] = []
    restored_count = 0
    restored_historical_split_counts: Counter[str] = Counter()
    restored_historical_split_mismatch_count = 0
    seen: set[str] = set()
    inputs: dict[str, dict[str, str]] = {"novel": {}, "brighter": {}}
    for kind, paths in (("novel", novel_paths), ("brighter", brighter_paths)):
        for split, path in _split_paths(paths):
            inputs[kind][split] = file_sha256(path)
            for legacy in _read_jsonl(path):
                candidate, restored, historical_audit_split = _candidate_from_legacy(
                    legacy, split, kind, audit
                )
                sample_id = str(candidate["id"])
                if sample_id in seen:
                    raise ValueError(f"候选 ID 重复: {sample_id}")
                seen.add(sample_id)
                restored_count += int(restored)
                if historical_audit_split is not None:
                    restored_historical_split_counts[historical_audit_split] += 1
                    restored_historical_split_mismatch_count += int(historical_audit_split != split)
                rows.append(candidate)
    rows.sort(key=lambda row: (_SPLITS.index(str(row["split"])), str(row["id"])))
    target = Path(output_dir)
    candidates_path = target / "candidates.jsonl"
    _atomic_jsonl(candidates_path, rows)
    report = {
        "schema": CANDIDATE_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "candidateCount": len(rows),
        "splitCounts": dict(Counter(str(row["split"]) for row in rows)),
        "sourceCounts": dict(Counter(str(row["sourceKind"]) for row in rows)),
        "restoredOriginalDirectionCount": restored_count,
        "restoredHistoricalAuditSplitCounts": dict(restored_historical_split_counts),
        "restoredHistoricalAuditSplitMismatchCount": restored_historical_split_mismatch_count,
        "brighterSentenceTypeRule": "paired-whole-target-quotes-v1",
        "inputs": inputs,
        "candidateFile": str(candidates_path.resolve()),
        "candidateFileSha256": file_sha256(candidates_path),
    }
    _atomic_json(target / "prepare-report.json", report)
    return report


def direction_sets(
    emotions: Mapping[str, object],
    observed_directions: Iterable[str] | None = None,
    *,
    threshold: float = SIGNIFICANT_THRESHOLD,
    dominance_tolerance: float = DOMINANCE_TOLERANCE,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = _as_emotions(emotions, "emotions")
    observed_set = set(EMOTION_NAMES if observed_directions is None else observed_directions)
    if not observed_set.issubset(EMOTION_NAMES):
        raise ValueError("observed_directions 包含未知情感")
    significant = tuple(name for name in EMOTION_NAMES if name in observed_set and values[name] >= threshold)
    if not significant:
        return (), ()
    maximum = max(values[name] for name in observed_set)
    dominant = tuple(name for name in significant if maximum - values[name] <= dominance_tolerance + 1e-12)
    return significant, dominant


def detect_conflicts(
    candidate: Mapping[str, object],
    qwen_emotions: Mapping[str, object],
    *,
    threshold: float = SIGNIFICANT_THRESHOLD,
    dominance_tolerance: float = DOMINANCE_TOLERANCE,
) -> tuple[str, ...]:
    source_kind = str(candidate.get("sourceKind", ""))
    observed = tuple(str(item) for item in candidate.get("observedDirections", []))
    old = tuple(str(item) for item in candidate.get("oldDirections", []))
    old_dominant = tuple(str(item) for item in candidate.get("oldDominantDirections", []))
    compare_observed = EMOTION_NAMES if source_kind == "novel" else observed
    significant, dominant = direction_sets(
        qwen_emotions,
        compare_observed,
        threshold=threshold,
        dominance_tolerance=dominance_tolerance,
    )
    # The legacy labels provide discrete direction evidence only.  If Qwen's
    # complete significant direction set agrees, a change in relative
    # continuous strength does not justify a second Agent label.
    if set(significant) == set(old):
        return ()
    reasons: list[str] = []
    if source_kind == "novel" and (not old) != (not significant):
        reasons.append("base-state-conflict")
    if any(name not in significant for name in old):
        reasons.append("old-direction-missing")
    if set(dominant) != set(old_dominant):
        reasons.append("dominant-direction-conflict")
    return tuple(reasons)


def _validated_candidates(path: str | Path) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", ""))
        try:
            schema_version = int(row.get("schemaVersion", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"候选 schemaVersion 无效: {sample_id}") from exc
        if row.get("schema") != CANDIDATE_SCHEMA or schema_version != 3:
            raise ValueError(f"候选 schema 无效: {sample_id}")
        supplied = str(row.get("inputSha256", ""))
        unhashed = dict(row)
        unhashed.pop("inputSha256", None)
        if supplied != _canonical_hash(unhashed):
            raise ValueError(f"候选输入哈希不匹配: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"候选 ID 重复: {sample_id}")
        seen.add(sample_id)
    return rows


def _validated_qwen(path: str | Path, candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    if len(rows) != len(candidates):
        raise ValueError("Qwen 标注未完整覆盖候选")
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        if row.get("schema") != QWEN_SCHEMA or sample_id in by_id:
            raise ValueError(f"Qwen schema 无效或 ID 重复: {sample_id}")
        normalized = dict(row)
        normalized["emotions"] = _as_emotions(row.get("emotions"), f"{sample_id}.qwenEmotions")
        raw_response = row.get("rawResponse")
        if not isinstance(raw_response, str):
            raise ValueError(f"Qwen 缺少原始响应: {sample_id}")
        parsed_raw = strict_parse_raw_emotions(raw_response)
        if parsed_raw != normalized["emotions"]:
            raise ValueError(f"Qwen 原始响应与解析向量不匹配: {sample_id}")
        if not str(row.get("modelFingerprint", "")) or not str(row.get("promptFingerprint", "")):
            raise ValueError(f"Qwen 模型或提示指纹缺失: {sample_id}")
        by_id[sample_id] = normalized
    ordered: list[dict[str, object]] = []
    for candidate in candidates:
        sample_id = str(candidate["id"])
        row = by_id.get(sample_id)
        if row is None:
            raise ValueError(f"Qwen 缺少候选: {sample_id}")
        for field in ("split", "inputSha256", "textSha256"):
            expected = candidate[field]
            actual = row.get(field)
            if actual != expected:
                raise ValueError(f"Qwen {field} 不匹配: {sample_id}")
        ordered.append(row)
    return ordered


def _review_queue_rows(
    candidates: Sequence[Mapping[str, object]], qwen_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    queue: list[dict[str, object]] = []
    for candidate, qwen in zip(candidates, qwen_rows):
        emotions = _as_emotions(qwen["emotions"], f"{candidate['id']}.qwenEmotions")
        reasons = detect_conflicts(candidate, emotions)
        if not reasons:
            continue
        row: dict[str, object] = {
            "schema": REVIEW_QUEUE_SCHEMA,
            "schemaVersion": SCHEMA_VERSION,
            "id": candidate["id"],
            "split": candidate["split"],
            "workId": candidate["workId"],
            "previousText": candidate["previousText"],
            "text": candidate["text"],
            "sentenceType": candidate["sentenceType"],
            "sourceKind": candidate["sourceKind"],
            "source": candidate["source"],
            "observedDirections": candidate["observedDirections"],
            "oldDirections": candidate["oldDirections"],
            "oldDominantDirections": candidate["oldDominantDirections"],
            "qwenEmotions": emotions,
            "conflictReasons": list(reasons),
            "candidateInputSha256": candidate["inputSha256"],
            "textSha256": candidate["textSha256"],
            "qwenModelFingerprint": qwen.get("modelFingerprint", ""),
            "qwenPromptFingerprint": qwen.get("promptFingerprint", ""),
        }
        row["inputSha256"] = _canonical_hash(row)
        queue.append(row)
    return queue


def prepare_continuous_v3_reviews(candidates_path, qwen_annotations_path, output_dir, chunk_size=50):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    candidates = _validated_candidates(candidates_path)
    qwen = _validated_qwen(qwen_annotations_path, candidates)
    queue = _review_queue_rows(candidates, qwen)
    target = Path(output_dir)
    queue_path = target / "review-queue.jsonl"
    _atomic_jsonl(queue_path, queue)
    chunks = []
    for start in range(0, len(queue), chunk_size):
        path = target / "chunks" / f"chunk-{start // chunk_size:04d}.jsonl"
        part = queue[start : start + chunk_size]
        _atomic_jsonl(path, part)
        chunks.append({"chunkId": start // chunk_size, "rows": len(part), "path": str(path.resolve()), "sha256": file_sha256(path)})
    report = {
        "schema": REVIEW_QUEUE_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "candidateCount": len(candidates),
        "reviewCount": len(queue),
        "chunkSize": chunk_size,
        "chunks": chunks,
        "candidateFileSha256": file_sha256(candidates_path),
        "qwenAnnotationFileSha256": file_sha256(qwen_annotations_path),
        "reviewQueueFile": str(queue_path.resolve()),
        "reviewQueueFileSha256": file_sha256(queue_path),
        "conflictReasons": dict(Counter(reason for row in queue for reason in row["conflictReasons"])),
    }
    _atomic_json(target / "review-manifest.json", report)
    return report


def _direction_mask(names: Iterable[str]) -> int:
    selected = set(names)
    return sum(1 << index for index, name in enumerate(EMOTION_NAMES) if name in selected)


def _comparison_profile(row: Mapping[str, object]) -> dict[str, object]:
    observed = EMOTION_NAMES if row.get("sourceKind") == "novel" else tuple(
        str(name) for name in row.get("observedDirections", [])
    )
    significant, dominant = direction_sets(row["qwenEmotions"], observed)
    old = tuple(str(name) for name in row.get("oldDirections", []))
    anchor = old or significant or dominant
    primary = next((name for name in EMOTION_NAMES if name in anchor), "base")
    values = _as_emotions(row["qwenEmotions"], f"{row.get('id', '')}.qwenEmotions")
    anchor_strength = max((values[name] for name in anchor), default=0.0)
    return {
        "anchorDirections": list(anchor),
        "primaryDirection": primary,
        "qwenSignificantDirections": list(significant),
        "qwenDominantDirections": list(dominant),
        "anchorStrength": anchor_strength,
    }


def _comparison_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    profile = _comparison_profile(row)
    anchor = tuple(str(name) for name in profile["anchorDirections"])
    significant = tuple(str(name) for name in profile["qwenSignificantDirections"])
    dominant = tuple(str(name) for name in profile["qwenDominantDirections"])
    primary = str(profile["primaryDirection"])
    primary_index = len(EMOTION_NAMES) if primary == "base" else EMOTION_NAMES.index(primary)
    return (
        primary_index,
        len(anchor),
        _direction_mask(anchor),
        _direction_mask(significant),
        _direction_mask(dominant),
        str(row.get("sourceKind", "")),
        str(row.get("sentenceType", "")),
        float(profile["anchorStrength"]),
        str(row.get("id", "")),
    )


def prepare_continuous_v3_comparison_batches(
    queue_path,
    output_dir,
    *,
    batch_size=100,
):
    """Regroup review rows for within-batch direction/intensity calibration."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 1:
        raise ValueError("batch_size 必须是大于 1 的整数")
    queue = _validated_queue(queue_path)
    ordered = sorted(queue, key=_comparison_sort_key)
    if {str(row["id"]) for row in ordered} != {str(row["id"]) for row in queue}:
        raise AssertionError("比较分批前后 ID 集合不一致")
    target = Path(output_dir)
    ordered_path = target / "review-queue.jsonl"
    _atomic_jsonl(ordered_path, ordered)
    batches: list[dict[str, object]] = []
    for start in range(0, len(ordered), batch_size):
        index = start // batch_size
        rows = ordered[start : start + batch_size]
        path = target / "chunks" / f"chunk-{index:04d}.jsonl"
        _atomic_jsonl(path, rows)
        profiles = [_comparison_profile(row) for row in rows]
        batches.append(
            {
                "batchId": index,
                "rows": len(rows),
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "primaryDirectionCounts": dict(
                    Counter(str(profile["primaryDirection"]) for profile in profiles)
                ),
                "anchorDirectionSignatures": dict(
                    Counter(
                        "+".join(str(name) for name in profile["anchorDirections"]) or "base"
                        for profile in profiles
                    )
                ),
            }
        )
    report = {
        "schema": REVIEW_QUEUE_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "groupingRule": (
            "legacy-directions-else-qwen-significant; primary-direction; "
            "anchor/significant/dominant masks; source; sentence-type; ascending-anchor-strength-v1"
        ),
        "reviewCount": len(ordered),
        "batchSize": batch_size,
        "batchCount": len(batches),
        "sourceQueueFileSha256": file_sha256(queue_path),
        "reviewQueueFile": str(ordered_path.resolve()),
        "reviewQueueFileSha256": file_sha256(ordered_path),
        "batches": batches,
    }
    _atomic_json(target / "comparison-batch-manifest.json", report)
    return report


def _validated_queue(path: str | Path) -> list[dict[str, object]]:
    rows = _read_jsonl(path, allow_empty=True)
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("id", ""))
        if row.get("schema") != REVIEW_QUEUE_SCHEMA or int(row.get("schemaVersion", -1)) != 3:
            raise ValueError(f"复核队列 schema 无效: {sample_id}")
        if not sample_id or sample_id in seen:
            raise ValueError(f"复核队列 ID 缺失或重复: {sample_id or '<empty>'}")
        unhashed = dict(row)
        supplied = str(unhashed.pop("inputSha256", ""))
        if supplied != _canonical_hash(unhashed):
            raise ValueError(f"复核队列输入哈希不匹配: {sample_id}")
        _as_emotions(row.get("qwenEmotions"), f"{sample_id}.qwenEmotions")
        seen.add(sample_id)
    return rows


def _validate_review_rows(
    queue: Sequence[Mapping[str, object]], reviews_path: str | Path, pass_name: str
) -> list[dict[str, object]]:
    reviews = _read_jsonl(reviews_path, allow_empty=True)
    if len(reviews) != len(queue):
        raise ValueError(f"{pass_name} 缺行：复核数 {len(reviews)} 不等于队列数 {len(queue)}")
    expected_fields = {
        "schema",
        "schemaVersion",
        "id",
        "inputSha256",
        "decision",
        "emotions",
        "confidence",
        "reason",
    }
    by_id: dict[str, dict[str, object]] = {}
    for review in reviews:
        sample_id = str(review.get("id", ""))
        if set(review) != expected_fields:
            raise ValueError(f"{pass_name} 输出字段不严格: {sample_id or '<empty>'}")
        if review.get("schema") != REVIEW_SCHEMA or int(review.get("schemaVersion", -1)) != 3:
            raise ValueError(f"{pass_name} schema 无效: {sample_id}")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"{pass_name} ID 缺失或重复: {sample_id or '<empty>'}")
        decision = str(review.get("decision", ""))
        if decision not in {"keep", "revise"}:
            raise ValueError(f"{pass_name} decision 必须为 keep/revise: {sample_id}")
        confidence = str(review.get("confidence", ""))
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"{pass_name} confidence 无效: {sample_id}")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{pass_name} reason 不能为空: {sample_id}")
        normalized = dict(review)
        normalized["emotions"] = _as_emotions(review.get("emotions"), f"{sample_id}.emotions")
        by_id[sample_id] = normalized
    validated: list[dict[str, object]] = []
    for queue_row in queue:
        sample_id = str(queue_row["id"])
        review = by_id.get(sample_id)
        if review is None:
            raise ValueError(f"{pass_name} 缺少 ID: {sample_id}")
        if review["inputSha256"] != queue_row["inputSha256"]:
            raise ValueError(f"{pass_name} 输入哈希不匹配: {sample_id}")
        if review["decision"] == "keep" and review["emotions"] != _as_emotions(
            queue_row["qwenEmotions"], f"{sample_id}.qwenEmotions"
        ):
            raise ValueError(f"{pass_name} keep 必须严格保留 Qwen 向量: {sample_id}")
        validated.append(review)
    return validated


def validate_continuous_v3_reviews(queue_path, reviews_path, pass_name="review"):
    """Strictly validate one independent agent pass against a queue."""
    queue = _validated_queue(queue_path)
    reviews = _validate_review_rows(queue, reviews_path, str(pass_name))
    return {
        "schema": REVIEW_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "passName": str(pass_name),
        "queueFileSha256": file_sha256(queue_path),
        "reviewFileSha256": file_sha256(reviews_path),
        "reviewCount": len(reviews),
        "keepCount": sum(row["decision"] == "keep" for row in reviews),
        "reviseCount": sum(row["decision"] == "revise" for row in reviews),
        "reviews": reviews,
    }


def assemble_continuous_v3_review_chunks(
    queue_path,
    reviews_dir,
    output_path,
    *,
    pass_name="review",
    chunk_size=50,
):
    """Validate every isolated review chunk and atomically assemble one pass."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    queue = _validated_queue(queue_path)
    directory = Path(reviews_dir)
    expected_names = {
        f"chunk-{index:04d}.jsonl"
        for index in range((len(queue) + chunk_size - 1) // chunk_size)
    }
    actual_paths = {path.name: path for path in directory.glob("chunk-*.jsonl") if path.is_file()}
    if set(actual_paths) != expected_names:
        missing = sorted(expected_names - set(actual_paths))
        extra = sorted(set(actual_paths) - expected_names)
        raise ValueError(f"{pass_name} 分片集合不完整，缺失={missing[:5]}，多余={extra[:5]}")
    assembled: list[dict[str, object]] = []
    for index in range((len(queue) + chunk_size - 1) // chunk_size):
        start = index * chunk_size
        part = queue[start : start + chunk_size]
        assembled.extend(
            _validate_review_rows(
                part,
                actual_paths[f"chunk-{index:04d}.jsonl"],
                f"{pass_name}/chunk-{index:04d}",
            )
        )
    target = Path(output_path)
    _atomic_jsonl(target, assembled)
    return {
        "schema": REVIEW_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "passName": str(pass_name),
        "queueFileSha256": file_sha256(queue_path),
        "reviewFile": str(target.resolve()),
        "reviewFileSha256": file_sha256(target),
        "reviewCount": len(assembled),
        "chunkCount": len(expected_names),
        "chunkSize": chunk_size,
        "keepCount": sum(row["decision"] == "keep" for row in assembled),
        "reviseCount": sum(row["decision"] == "revise" for row in assembled),
    }


def _validated_single_reviews(
    path: str | Path,
    candidates: Sequence[Mapping[str, object]],
    qwen: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Validate one Agent pass and adapt it to the finalization review shape."""
    expected_queue = _review_queue_rows(candidates, qwen)
    reviews = _validate_review_rows(expected_queue, path, "single-pass")
    materialized: list[dict[str, object]] = []
    for queue_row, review in zip(expected_queue, reviews):
        materialized.append(
            {
                "schema": REVIEW_SCHEMA,
                "schemaVersion": SCHEMA_VERSION,
                "id": queue_row["id"],
                "candidateInputSha256": queue_row["candidateInputSha256"],
                "reviewInputSha256": queue_row["inputSha256"],
                "decision": review["decision"],
                "emotions": _as_emotions(
                    review["emotions"], f"{queue_row['id']}.singlePassEmotions"
                ),
                "confidence": review["confidence"],
                "reason": review["reason"],
                "resolution": "single-agent",
            }
        )
    return materialized, expected_queue, reviews


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _dataset_statistics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    per_emotion: dict[str, object] = {}
    for name in EMOTION_NAMES:
        values = [float(row["emotions"][name]) for row in rows]  # type: ignore[index]
        per_emotion[name] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "quantiles": {
                key: _quantile(values, fraction)
                for key, fraction in (("p0", 0.0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p100", 1.0))
            },
            "nonZeroRate": sum(value > 0.0 for value in values) / len(values) if values else 0.0,
        }
    dominant: Counter[str] = Counter()
    for row in rows:
        values = _as_emotions(row["emotions"], f"{row['id']}.emotions")
        maximum = max(values.values())
        key = "base" if maximum == 0.0 else "+".join(name for name in EMOTION_NAMES if values[name] == maximum)
        dominant[key] += 1
    return {
        "perEmotion": per_emotion,
        "allZeroRate": sum(not any(row["emotions"].values()) for row in rows) / len(rows) if rows else 0.0,  # type: ignore[union-attr]
        "dominantEmotion": dict(dominant),
        "sentenceTypes": dict(Counter(str(row["sentenceType"]) for row in rows)),
        "sources": dict(Counter(str(row["sourceKind"]) for row in rows)),
        "sourceDetails": dict(Counter(str(row["source"]) for row in rows)),
    }


def _validated_exclusions(
    path: str | Path | None,
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if path is None:
        return []
    rows = _read_jsonl(path)
    candidate_ids = {str(row["id"]) for row in candidates}
    seen: set[str] = set()
    expected_fields = {
        "schema",
        "schemaVersion",
        "id",
        "split",
        "sourceKind",
        "candidateInputSha256",
        "textSha256",
        "modelFingerprint",
        "promptFingerprint",
        "attempts",
        "reason",
    }
    for row in rows:
        sample_id = str(row.get("id", ""))
        if set(row) != expected_fields:
            raise ValueError(f"Qwen 排除审计字段不严格: {sample_id or '<empty>'}")
        if row.get("schema") != QWEN_EXCLUSION_SCHEMA or row.get("schemaVersion") != 3:
            raise ValueError(f"Qwen 排除审计 schema 无效: {sample_id}")
        if not sample_id or sample_id in seen or sample_id in candidate_ids:
            raise ValueError(f"Qwen 排除审计 ID 缺失、重复或仍在有效候选中: {sample_id}")
        if row.get("split") not in _SPLITS or row.get("sourceKind") not in {"novel", "brighter"}:
            raise ValueError(f"Qwen 排除审计 split/source 无效: {sample_id}")
        for field in (
            "candidateInputSha256",
            "textSha256",
            "modelFingerprint",
            "promptFingerprint",
        ):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Qwen 排除审计 {field} 无效: {sample_id}")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 3:
            raise ValueError(f"Qwen 排除审计必须完整保留三次尝试: {sample_id}")
        for index, attempt in enumerate(attempts, 1):
            if (
                not isinstance(attempt, dict)
                or set(attempt) != {"attempt", "rawResponse", "error"}
                or attempt.get("attempt") != index
                or not isinstance(attempt.get("rawResponse"), str)
                or not isinstance(attempt.get("error"), str)
                or not attempt["error"]
            ):
                raise ValueError(f"Qwen 排除审计第 {index} 次尝试无效: {sample_id}")
        if row.get("reason") != "strict-qwen-output-failed-after-three-attempts":
            raise ValueError(f"Qwen 排除原因无效: {sample_id}")
        seen.add(sample_id)
    return rows


def finalize_continuous_v3(
    candidates_path,
    qwen_annotations_path,
    reviews_path,
    output_dir,
    expected_split_counts=None,
    exclusions_path=None,
):
    """Materialize v3 directly from one strictly validated Codex Agent pass."""
    expected_counts = dict(DEFAULT_SPLIT_COUNTS if expected_split_counts is None else expected_split_counts)
    if set(expected_counts) != set(_SPLITS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in expected_counts.values()
    ):
        raise ValueError("expected_split_counts 必须包含非负整数 train/dev/test")
    candidates = _validated_candidates(candidates_path)
    qwen = _validated_qwen(qwen_annotations_path, candidates)
    exclusions = _validated_exclusions(exclusions_path, candidates)
    reviewed, conflict_queue, review_audit = _validated_single_reviews(
        reviews_path, candidates, qwen
    )
    reviewed_by_id = {str(row["id"]): row for row in reviewed}
    qwen_by_id = {str(row["id"]): row for row in qwen}

    split_counts = Counter(str(row["split"]) for row in candidates)
    if dict(split_counts) != expected_counts:
        raise ValueError(f"最终 split 计数不匹配: actual={dict(split_counts)}, expected={expected_counts}")
    owners: dict[str, str] = {}
    for candidate in candidates:
        work_id = str(candidate["workId"])
        split = str(candidate["split"])
        previous = owners.setdefault(work_id, split)
        if previous != split:
            raise ValueError(f"作品级 split 泄漏: {work_id} 同时出现在 {previous}/{split}")

    outputs: dict[str, list[dict[str, object]]] = {split: [] for split in _SPLITS}
    sources: list[dict[str, object]] = []
    for candidate in candidates:
        sample_id = str(candidate["id"])
        qwen_row = qwen_by_id[sample_id]
        review = reviewed_by_id.get(sample_id)
        emotions = _as_emotions(
            review["emotions"] if review is not None else qwen_row["emotions"],
            f"{sample_id}.finalEmotions",
        )
        final = {
            "schema": FINAL_SCHEMA,
            "schemaVersion": SCHEMA_VERSION,
            "id": sample_id,
            "split": candidate["split"],
            "workId": candidate["workId"],
            "previousText": candidate["previousText"],
            "text": candidate["text"],
            "sentenceType": candidate["sentenceType"],
            "sourceKind": candidate["sourceKind"],
            "source": candidate["source"],
            "licenseId": candidate["licenseId"],
            "licenseStatus": candidate["licenseStatus"],
            "emotions": emotions,
        }
        if {"intensity", "primaryEmotion", "labelMask"}.intersection(final):
            raise AssertionError("v3 最终 schema 意外包含旧标签字段")
        outputs[str(candidate["split"])].append(final)
        sources.append(
            {
                "id": sample_id,
                "split": candidate["split"],
                "candidateInputSha256": candidate["inputSha256"],
                "qwenEmotions": _as_emotions(qwen_row["emotions"], f"{sample_id}.qwenEmotions"),
                "finalEmotions": emotions,
                "labelSource": "qwen" if review is None or review["decision"] == "keep" else "agent",
                "decision": "keep" if review is None else review["decision"],
                "reviewInputSha256": review.get("reviewInputSha256") if review is not None else None,
                "reviewResolution": review.get("resolution") if review is not None else None,
                "qwenModelFingerprint": qwen_row.get("modelFingerprint", ""),
                "qwenPromptFingerprint": qwen_row.get("promptFingerprint", ""),
            }
        )

    target = Path(output_dir)
    for split in _SPLITS:
        outputs[split].sort(key=lambda row: str(row["id"]))
        _atomic_jsonl(target / f"{split}.jsonl", outputs[split])
    audit_dir = target / "audit"
    audit_files = {
        "candidates.jsonl": candidates,
        "qwen-annotations.jsonl": qwen,
        "review-queue.jsonl": conflict_queue,
        "agent-reviews.jsonl": review_audit,
        "final-sources.jsonl": sources,
    }
    if exclusions:
        audit_files["excluded-invalid.jsonl"] = exclusions
    for name, rows in audit_files.items():
        _atomic_jsonl(audit_dir / name, rows)
    all_rows = [row for split in _SPLITS for row in outputs[split]]
    reason_counts = Counter(reason for row in conflict_queue for reason in row["conflictReasons"])
    resolution_counts = Counter(str(row.get("resolution", "")) for row in reviewed)
    revised_count = sum(row["decision"] == "revise" for row in reviewed)
    hash_paths = [target / f"{split}.jsonl" for split in _SPLITS] + [audit_dir / name for name in audit_files]
    hashes = {str(path.relative_to(target)).replace("\\", "/"): file_sha256(path) for path in hash_paths}
    report = {
        "schema": FINAL_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "recordCount": len(all_rows),
        "splitCounts": dict(split_counts),
        "workCount": len(owners),
        "exclusions": {
            "count": len(exclusions),
            "splitCounts": dict(Counter(str(row["split"]) for row in exclusions)),
            "sourceCounts": dict(Counter(str(row["sourceKind"]) for row in exclusions)),
            "originalCandidateCount": len(all_rows) + len(exclusions),
        },
        "statistics": _dataset_statistics(all_rows),
        "review": {
            "mode": "single-agent",
            "conflictCount": len(conflict_queue),
            "conflictReasons": dict(reason_counts),
            "resolutionCounts": dict(resolution_counts),
            "qwenRetainedCount": len(all_rows) - revised_count,
            "agentRevisedCount": revised_count,
        },
        "fileSha256": hashes,
    }
    report_path = target / "report.json"
    _atomic_json(report_path, report)
    manifest = {
        "schema": FINAL_SCHEMA,
        "schemaVersion": SCHEMA_VERSION,
        "status": "completed",
        "recordCount": len(all_rows),
        "splitCounts": dict(split_counts),
        "candidateFileSha256": file_sha256(candidates_path),
        "qwenAnnotationFileSha256": file_sha256(qwen_annotations_path),
        "reviewMode": "single-agent",
        "reviewFileSha256": file_sha256(reviews_path),
        "exclusionFileSha256": file_sha256(exclusions_path) if exclusions_path is not None else None,
        "reportFileSha256": file_sha256(report_path),
        "fileSha256": hashes,
        "qwenPrompt": "system=文本情感分类; user=TARGET_TEXT_ONLY; do_sample=False",
        "labelPolicy": "raw-qwen-continuous-plus-single-agent-continuous-correction",
        "allZeroMeaning": "neutral/base; downstream TTS uses voice base_emotion",
        "significantThreshold": SIGNIFICANT_THRESHOLD,
        "dominanceTolerance": DOMINANCE_TOLERANCE,
    }
    _atomic_json(target / "manifest.json", manifest)
    return report
