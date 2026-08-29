"""Prepare a targeted annotation backlog for underrepresented emotions."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .annotation import file_sha256
from .source_extract import (
    CANDIDATE_SCHEMA,
    _candidate_record,
    _previous_sentence,
    normalize_text,
    sha256_bytes,
)


COVERAGE_SCHEMA = "readest-emotion-coverage-candidate-v1"
TARGET_PATTERNS = {
    "disgusted": re.compile(
        r"厌恶|恶心|反胃|作呕|嫌恶|鄙夷|不屑|唾弃|讨厌|憎恶|排斥|"
        r"反感|肮脏|恶臭|鄙视|可憎|令人作呕"
    ),
    "melancholic": re.compile(
        r"忧郁|惆怅|落寞|寂寞|孤独|黯然|低落|消沉|萧瑟|怅然|"
        r"哀愁|心灰意冷|无精打采|郁郁|凄凉|悲凉|愁绪|孤寂|落寂"
    ),
}


def _existing_ids(path: str | Path) -> set[str]:
    result: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not str(value.get("id", "")):
                raise ValueError(f"现有候选第 {line_number} 行无有效 id")
            candidate_id = str(value["id"])
            if candidate_id in result:
                raise ValueError(f"现有候选含重复 id: {candidate_id}")
            result.add(candidate_id)
    return result


def _balanced_select(
    rows: list[dict[str, object]], quota: int, works: list[str]
) -> list[dict[str, object]]:
    by_work: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_work[str(row["workId"])].append(row)
    for items in by_work.values():
        items.sort(key=lambda row: (-int(row["coverageScore"]), str(row["id"])))
    selected: list[dict[str, object]] = []
    per_work = max(1, math.ceil(quota / max(1, len(works))))
    for work in works:
        selected.extend(by_work.get(work, [])[:per_work])
    selected_ids = {str(row["id"]) for row in selected}
    if len(selected) < quota:
        remaining = sorted(
            (row for row in rows if str(row["id"]) not in selected_ids),
            key=lambda row: (-int(row["coverageScore"]), str(row["id"])),
        )
        selected.extend(remaining[: quota - len(selected)])
    return selected[:quota]


def build_coverage_backlog(
    input_root: str | Path,
    existing_candidates: str | Path,
    output_dir: str | Path,
    *,
    quotas: Mapping[str, int],
    license_status: str = "test-only",
) -> dict[str, object]:
    if license_status not in {"pending", "test-only"}:
        raise ValueError("license_status 只能是 pending 或 test-only")
    unknown = set(quotas).difference(TARGET_PATTERNS)
    if unknown:
        raise ValueError(f"不支持的覆盖目标: {sorted(unknown)}")
    normalized_quotas = {name: max(0, int(quotas.get(name, 0))) for name in TARGET_PATTERNS}
    if not any(normalized_quotas.values()):
        raise ValueError("至少需要一个正数覆盖配额")

    root = Path(input_root)
    files = sorted(root.glob("*_annotated.jsonl"))
    if not files:
        raise ValueError(f"未找到 *_annotated.jsonl: {root}")
    works = [path.stem.removesuffix("_annotated") for path in files]
    excluded_ids = _existing_ids(existing_candidates)
    by_emotion: dict[str, list[dict[str, object]]] = {
        name: [] for name in TARGET_PATTERNS
    }
    source_manifest: list[dict[str, object]] = []
    scanned_windows = 0
    scanned_sentences = 0

    for source_path in files:
        source_hash = sha256_bytes(source_path.read_bytes())
        work_id = source_path.stem.removesuffix("_annotated")
        source_manifest.append(
            {
                "workId": work_id,
                "sourceFile": source_path.name,
                "sourceSha256": source_hash,
                "licenseStatus": license_status,
            }
        )
        with source_path.open("r", encoding="utf-8") as handle:
            for source_row, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    continue
                scanned_windows += 1
                sentences = value.get("sentences")
                if not isinstance(sentences, list):
                    continue
                sentence_list = [item for item in sentences if isinstance(item, dict)]
                scanned_sentences += len(sentence_list)
                target_ids: set[int] = set()
                for target_id in value.get("targets", []):
                    try:
                        target_ids.add(int(target_id))
                    except (TypeError, ValueError):
                        continue
                for index, sentence in enumerate(sentence_list):
                    try:
                        sentence_id = int(sentence.get("sentenceId"))
                    except (TypeError, ValueError):
                        continue
                    sentence_type = normalize_text(sentence.get("type"))
                    is_target = sentence_id in target_ids
                    if not (
                        (is_target and sentence_type == "dialogue")
                        or (not is_target and sentence_type == "narration")
                    ):
                        continue
                    previous_text = _previous_sentence(sentence_list, index)
                    candidate = _candidate_record(
                        source_path=source_path,
                        source_sha256=source_hash,
                        work_id=work_id,
                        source_row=source_row,
                        sentence=sentence,
                        previous_text=previous_text,
                        is_dialogue_target=is_target,
                    )
                    if candidate is None:
                        continue
                    _, record = candidate
                    if str(record["id"]) in excluded_ids:
                        continue
                    text = str(record["text"])
                    previous = str(record.get("previousText", ""))
                    for emotion, pattern in TARGET_PATTERNS.items():
                        current_hits = pattern.findall(text)
                        previous_hits = pattern.findall(previous)
                        if not current_hits and not previous_hits:
                            continue
                        item = dict(record)
                        item.update(
                            {
                                "schema": COVERAGE_SCHEMA,
                                "licenseStatus": license_status,
                                "coverageTarget": emotion,
                                "coverageEvidence": {
                                    "currentHits": sorted(set(current_hits)),
                                    "previousHits": sorted(set(previous_hits)),
                                },
                                "coverageScore": len(current_hits) * 100
                                + len(previous_hits) * 20
                                + min(19, len(text) // 8),
                            }
                        )
                        by_emotion[emotion].append(item)

    selected_by_emotion = {
        emotion: _balanced_select(by_emotion[emotion], normalized_quotas[emotion], works)
        for emotion in TARGET_PATTERNS
        if normalized_quotas[emotion]
    }
    merged: dict[str, dict[str, object]] = {}
    for emotion, rows in selected_by_emotion.items():
        for row in rows:
            candidate_id = str(row["id"])
            existing = merged.get(candidate_id)
            if existing is None:
                record = dict(row)
                record["coverageTargets"] = [emotion]
                record.pop("coverageTarget", None)
                merged[candidate_id] = record
            else:
                targets = existing["coverageTargets"]
                assert isinstance(targets, list)
                if emotion not in targets:
                    targets.append(emotion)
                    targets.sort()

    records = [merged[key] for key in sorted(merged)]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "coverage-candidates.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    annotation_path = target / "annotation-candidates.jsonl"
    with annotation_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            blind = {
                key: value
                for key, value in record.items()
                if key not in {"coverageTargets", "coverageEvidence", "coverageScore"}
            }
            blind["schema"] = CANDIDATE_SCHEMA
            handle.write(json.dumps(blind, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = target / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": COVERAGE_SCHEMA,
                "licenseStatus": license_status,
                "sources": source_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    per_work = Counter(str(row["workId"]) for row in records)
    per_type = Counter(str(row["sentenceType"]) for row in records)
    report: dict[str, object] = {
        "schema": COVERAGE_SCHEMA,
        "licenseStatus": license_status,
        "scannedWindows": scanned_windows,
        "scannedSentences": scanned_sentences,
        "sourceCount": len(files),
        "existingCandidateCount": len(excluded_ids),
        "existingCandidatesSha256": file_sha256(existing_candidates),
        "availableMatches": {name: len(rows) for name, rows in by_emotion.items()},
        "requestedQuotas": normalized_quotas,
        "selectedByEmotion": {
            name: len(rows) for name, rows in selected_by_emotion.items()
        },
        "uniqueSelected": len(records),
        "workCoverage": dict(sorted(per_work.items())),
        "sentenceTypes": dict(sorted(per_type.items())),
        "outputSha256": file_sha256(output_path),
        "annotationInputSha256": file_sha256(annotation_path),
        "annotationInputBlinded": True,
        "sourceManifestSha256": file_sha256(manifest_path),
        "annotationStatus": "unlabeled-coverage-backlog",
    }
    (target / "coverage-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
