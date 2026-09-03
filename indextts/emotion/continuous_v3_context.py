"""Expand the r2 novel snapshot to three complete preceding sentences."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .context_policy import (
    LEGACY_CONTEXT_DELIMITER as CONTEXT_DELIMITER,
    LEGACY_CONTEXT_POLICY as CONTEXT_POLICY,
    LEGACY_CONTEXT_SENTENCE_LIMIT as CONTEXT_SENTENCE_LIMIT,
    select_complete_context,
)
from .dataset import format_current_text
from .schema import EMOTION_NAMES
from .source_extract import normalize_text


OUTPUT_SCHEMA = "readest-emotion-continuous-context-snapshot-v1"
AUDIT_SCHEMA = "readest-emotion-continuous-context-audit-v1"
EXPECTED_NOVEL_SPLITS = {"train": 13_496, "dev": 2_896, "test": 4_126}
EXPECTED_CONTEXT_COUNTS = {"0": 522, "1": 790, "2": 973, "3": 18_233}
EXPECTED_TARGET_TRUNCATED = 10
SNAPSHOT_MAX_LENGTH = 256

_ID = re.compile(r"^emotion-candidate-.+-r(?P<row>\d{6})-s(?P<sentence>\d{6})$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(record: Mapping[str, object]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocabulary": sorted(
            (str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()
        ),
        "specialTokens": {
            str(name): str(value)
            for name, value in sorted(tokenizer.special_tokens_map.items())
        },
    }
    return _canonical_sha256(payload)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败 {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 行必须是对象: {path}:{line_number}")
            yield line_number, record


def _load_novel_records(
    input_dir: Path,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, str]]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"缺少基线 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "readest-emotion-continuous-v3"
        or manifest.get("revision") != "r2"
    ):
        raise ValueError("输入目录不是预期的 r2 连续八维数据快照")

    records_by_split: dict[str, list[dict[str, object]]] = {}
    input_hashes = {"manifest.json": _sha256_file(manifest_path)}
    seen_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        path = input_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"缺少基线 split: {path}")
        expected_hash = (manifest.get("fileSha256") or {}).get(f"{split}.jsonl")
        actual_hash = _sha256_file(path)
        if expected_hash != actual_hash:
            raise ValueError(f"基线 split 哈希不匹配: {path}")
        input_hashes[f"{split}.jsonl"] = actual_hash
        novel_records: list[dict[str, object]] = []
        for line_number, record in _iter_jsonl(path):
            if record.get("sourceKind") != "novel":
                continue
            if record.get("schema") != "readest-emotion-continuous-v3":
                raise ValueError(f"schema 无效: {path}:{line_number}")
            if record.get("split") != split:
                raise ValueError(f"split 字段不匹配: {path}:{line_number}")
            record_id = str(record.get("id", ""))
            if not record_id or record_id in seen_ids:
                raise ValueError(f"小说记录 ID 为空或重复: {record_id!r}")
            seen_ids.add(record_id)
            emotions = record.get("emotions")
            if not isinstance(emotions, dict) or tuple(emotions) != EMOTION_NAMES:
                raise ValueError(f"八维 emotions 键序无效: {record_id}")
            novel_records.append(record)
        records_by_split[split] = novel_records
    actual_counts = {split: len(records) for split, records in records_by_split.items()}
    if actual_counts != EXPECTED_NOVEL_SPLITS:
        raise ValueError(
            f"r2 小说 split 计数不匹配: expected={EXPECTED_NOVEL_SPLITS}, actual={actual_counts}"
        )
    return records_by_split, input_hashes


def _record_location(record: Mapping[str, object]) -> tuple[str, int, int]:
    record_id = str(record.get("id", ""))
    match = _ID.fullmatch(record_id)
    if match is None:
        raise ValueError(f"无法从 ID 恢复源位置: {record_id}")
    source = str(record.get("source", ""))
    prefix = "speaker-id:"
    if not source.startswith(prefix):
        raise ValueError(f"小说 source 无效: {record_id}")
    source_stem = source.removeprefix(prefix)
    return source_stem, int(match.group("row")), int(match.group("sentence"))


def _load_source_rows(
    records_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    source_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, dict[str, object]]]:
    requested: dict[str, set[int]] = defaultdict(set)
    for records in records_by_split.values():
        for record in records:
            source_stem, row_index, _ = _record_location(record)
            requested[source_stem].add(row_index)

    rows: dict[tuple[str, int], dict[str, object]] = {}
    sources: dict[str, dict[str, object]] = {}
    for source_stem, row_indexes in sorted(requested.items()):
        path = source_root / f"{source_stem}.jsonl"
        if not path.is_file():
            raise ValueError(f"缺少原始小说标注文件: {path}")
        sources[source_stem] = {
            "path": path.resolve().as_posix(),
            "sha256": _sha256_file(path),
            "requestedRows": len(row_indexes),
        }
        remaining = set(row_indexes)
        for line_number, record in _iter_jsonl(path):
            if line_number in remaining:
                rows[(source_stem, line_number)] = record
                remaining.remove(line_number)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"原始标注缺少行 {sorted(remaining)[:5]}: {path}")
    return rows, sources


def _pair_token_count(tokenizer, previous_text: str, current_text: str) -> int:
    encoded = tokenizer(
        previous_text,
        current_text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def _expand_record(
    record: Mapping[str, object],
    source_row: Mapping[str, object],
    tokenizer,
    *,
    max_length: int,
) -> tuple[dict[str, object], dict[str, object]]:
    record_id = str(record["id"])
    _, source_row_index, source_sentence_id = _record_location(record)
    sentences = source_row.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"源窗口 sentences 无效: {record_id}")
    matches = [
        (index, sentence)
        for index, sentence in enumerate(sentences)
        if isinstance(sentence, dict)
        and str(sentence.get("sentenceId", "")) == str(source_sentence_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"源窗口目标 sentenceId 不是唯一匹配: {record_id}")
    target_index, target_sentence = matches[0]
    source_text = normalize_text(target_sentence.get("text"))
    source_type = normalize_text(target_sentence.get("type"))
    if source_text != record.get("text") or source_type != record.get("sentenceType"):
        raise ValueError(f"源窗口目标文本或句型不匹配: {record_id}")

    previous: list[tuple[int, str]] = []
    for sentence in sentences[:target_index]:
        if not isinstance(sentence, dict):
            continue
        text = normalize_text(sentence.get("text"))
        if not text:
            continue
        try:
            sentence_id = int(sentence.get("sentenceId"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"源窗口前句 sentenceId 无效: {record_id}") from exc
        previous.append((sentence_id, text))
    previous = previous[-CONTEXT_SENTENCE_LIMIT:]
    current_text = format_current_text(str(record["text"]), str(record["sentenceType"]))
    selection = select_complete_context(
        [text for _, text in previous],
        lambda previous_text: _pair_token_count(tokenizer, previous_text, current_text),
        max_length=max_length,
    )
    selected_count = len(selection.previous_sentences)
    selected_ids = [sentence_id for sentence_id, _ in previous[-selected_count:]] if selected_count else []
    output = dict(record)
    output["previousText"] = selection.previous_text
    audit = {
        "schema": AUDIT_SCHEMA,
        "schemaVersion": 1,
        "id": record_id,
        "split": record["split"],
        "workId": record["workId"],
        "sourceFile": f"{_record_location(record)[0]}.jsonl",
        "sourceRowIndex": source_row_index,
        "sourceSentenceId": source_sentence_id,
        "previousSentenceIds": selected_ids,
        "originalPreviousText": record.get("previousText", ""),
        "expandedPreviousText": selection.previous_text,
        "contextSentenceCount": selected_count,
        "pairTokenCount": selection.input_token_count,
        "targetTokenCount": selection.target_token_count,
        "stopReason": selection.stop_reason,
        "contextLimited": selection.context_limited,
        "targetTruncated": selection.target_truncated,
        "inputSha256": _canonical_sha256(record),
        "outputSha256": _canonical_sha256(output),
    }
    return output, audit


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def expand_continuous_v3_context(
    input_dir: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | Path = "hfl/chinese-macbert-base",
    max_length: int = SNAPSHOT_MAX_LENGTH,
    verify_expected_distribution: bool = True,
) -> dict[str, object]:
    """Materialize a novel-only r2 snapshot with up to three complete contexts."""

    if max_length != SNAPSHOT_MAX_LENGTH:
        raise ValueError(f"当前已物化快照固定 max_length={SNAPSHOT_MAX_LENGTH}")
    source_dir = Path(source_root)
    target = Path(output_dir)
    if target.exists():
        raise ValueError(f"输出目录已存在，拒绝覆盖: {target}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("扩充上文需要本地 transformers tokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), local_files_only=True, use_fast=True
    )
    tokenizer_sha256 = _tokenizer_fingerprint(tokenizer)
    records_by_split, input_hashes = _load_novel_records(Path(input_dir))
    source_rows, source_manifest = _load_source_rows(records_by_split, source_dir)

    expanded_by_split: dict[str, list[dict[str, object]]] = {}
    audits: list[dict[str, object]] = []
    context_counts: Counter[str] = Counter()
    stop_reasons: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        expanded: list[dict[str, object]] = []
        for record in records_by_split[split]:
            source_stem, source_row_index, _ = _record_location(record)
            output, audit = _expand_record(
                record,
                source_rows[(source_stem, source_row_index)],
                tokenizer,
                max_length=max_length,
            )
            audit["sourceFileSha256"] = source_manifest[source_stem]["sha256"]
            if output["emotions"] != record["emotions"]:
                raise AssertionError(f"扩充上文意外改变 emotions: {record['id']}")
            expanded.append(output)
            audits.append(audit)
            context_counts[str(audit["contextSentenceCount"])] += 1
            stop_reasons[str(audit["stopReason"])] += 1
        expanded_by_split[split] = expanded

    target_truncated = sum(bool(item["targetTruncated"]) for item in audits)
    if verify_expected_distribution:
        if dict(sorted(context_counts.items())) != EXPECTED_CONTEXT_COUNTS:
            raise ValueError(
                "上下文句数分布与锁定基线不一致: "
                f"expected={EXPECTED_CONTEXT_COUNTS}, actual={dict(context_counts)}"
            )
        if target_truncated != EXPECTED_TARGET_TRUNCATED:
            raise ValueError(
                f"超长目标数不一致: expected={EXPECTED_TARGET_TRUNCATED}, actual={target_truncated}"
            )

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        audit_dir = temp / "audit"
        audit_dir.mkdir(parents=True)
        for split, records in expanded_by_split.items():
            _write_jsonl(temp / f"{split}.jsonl", records)
        _write_jsonl(audit_dir / "context-expansion.jsonl", audits)
        _write_jsonl(
            audit_dir / "target-truncation-exceptions.jsonl",
            (item for item in audits if item["targetTruncated"]),
        )
        report = {
            "schema": OUTPUT_SCHEMA,
            "schemaVersion": 1,
            "recordCount": sum(len(records) for records in expanded_by_split.values()),
            "splitCounts": {split: len(records) for split, records in expanded_by_split.items()},
            "contextPolicy": CONTEXT_POLICY,
            "contextSentenceLimit": CONTEXT_SENTENCE_LIMIT,
            "contextDelimiter": CONTEXT_DELIMITER,
            "maxLength": max_length,
            "tokenizerSha256": tokenizer_sha256,
            "contextSentenceCounts": dict(sorted(context_counts.items())),
            "stopReasons": dict(sorted(stop_reasons.items())),
            "targetTruncatedCount": target_truncated,
            "labelsUnchanged": True,
            "trainingStarted": False,
        }
        (temp / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        material_files = [
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "report.json",
            "audit/context-expansion.jsonl",
            "audit/target-truncation-exceptions.jsonl",
        ]
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "schemaVersion": 1,
            "status": "completed",
            "parentDataset": Path(input_dir).resolve().as_posix(),
            "inputSha256": input_hashes,
            "sourceWindows": source_manifest,
            "baseModel": str(base_model),
            "tokenizerSha256": tokenizer_sha256,
            "tokenizerVocabularySize": len(tokenizer.get_vocab()),
            "contextPolicy": CONTEXT_POLICY,
            "contextSentenceLimit": CONTEXT_SENTENCE_LIMIT,
            "contextDelimiter": CONTEXT_DELIMITER,
            "maxLength": max_length,
            "splitCounts": report["splitCounts"],
            "recordCount": report["recordCount"],
            "fileSha256": {name: _sha256_file(temp / name) for name in material_files},
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return report | {
        "output": target.resolve().as_posix(),
        "manifestSha256": _sha256_file(target / "manifest.json"),
    }
