"""Versioned training-data schema for the Readest emotion classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
CONTINUOUS_SCHEMA_VERSION = 3
CONTINUOUS_SCHEMA = "readest-emotion-continuous-v3"
EMOTION_NAMES = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
SENTENCE_TYPES = frozenset({"narration", "dialogue"})
COMMERCIAL_LICENSES = frozenset(
    {
        "CC-BY-4.0",
        "CC0-1.0",
        "PUBLIC-DOMAIN",
        "PROPRIETARY-AUTHORIZED",
    }
)
LICENSE_STATUSES = frozenset({"approved", "pending", "test-only"})


@dataclass(frozen=True)
class EmotionExample:
    example_id: str
    work_id: str
    previous_text: str
    text: str
    sentence_type: str
    labels: tuple[float, ...]
    label_mask: tuple[float, ...]
    intensity: float
    license_id: str
    source: str
    license_status: str = "approved"
    annotation_status: str = ""
    adjudication_chosen_source: str | None = None
    adjudication_confidence: str | None = None

    def as_json(self) -> dict[str, object]:
        record = {
            "schemaVersion": SCHEMA_VERSION,
            "id": self.example_id,
            "workId": self.work_id,
            "previousText": self.previous_text,
            "text": self.text,
            "sentenceType": self.sentence_type,
            "emotions": dict(zip(EMOTION_NAMES, self.labels)),
            "labelMask": dict(zip(EMOTION_NAMES, self.label_mask)),
            "intensity": self.intensity,
            "licenseId": self.license_id,
            "licenseStatus": self.license_status,
            "source": self.source,
        }
        if self.annotation_status:
            record["annotationStatus"] = self.annotation_status
        if self.adjudication_chosen_source is not None:
            record["adjudicationChosenSource"] = self.adjudication_chosen_source
        if self.adjudication_confidence is not None:
            record["adjudicationConfidence"] = self.adjudication_confidence
        return record


def _bounded_number(value: object, field: str, *, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是数值") from exc
    if not 0.0 <= number <= maximum:
        raise ValueError(f"{field} 必须位于 [0, {maximum:g}]")
    return number


def _emotion_values(
    value: object,
    field: str,
    *,
    maximum: float = 1.0,
    default: float | None = None,
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        unknown = set(value).difference(EMOTION_NAMES)
        if unknown:
            raise ValueError(f"{field} 包含未知情感维度: {sorted(unknown)}")
        values = []
        for name in EMOTION_NAMES:
            if name not in value:
                if default is None:
                    raise ValueError(f"{field} 缺少 {name}")
                values.append(default)
            else:
                values.append(_bounded_number(value[name], f"{field}.{name}", maximum=maximum))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(EMOTION_NAMES):
            raise ValueError(f"{field} 必须正好包含 8 个值")
        return tuple(
            _bounded_number(item, f"{field}[{index}]", maximum=maximum)
            for index, item in enumerate(value)
        )
    raise ValueError(f"{field} 必须是对象或 8 维数组")


def parse_example(record: Mapping[str, object], *, origin: str = "<memory>") -> EmotionExample:
    schema_version = int(record.get("schemaVersion", SCHEMA_VERSION))
    if schema_version not in {SCHEMA_VERSION, CONTINUOUS_SCHEMA_VERSION}:
        raise ValueError(f"不支持的 schemaVersion={schema_version}: {origin}")
    is_continuous_v3 = schema_version == CONTINUOUS_SCHEMA_VERSION
    if is_continuous_v3:
        if record.get("schema") != CONTINUOUS_SCHEMA:
            raise ValueError(f"连续 v3 schema 无效: {origin}")
        legacy_labels = {"labelMask", "intensity", "primaryEmotion"}.intersection(record)
        if legacy_labels:
            raise ValueError(
                f"连续 v3 不得包含旧标签字段 {sorted(legacy_labels)}: {origin}"
            )

    example_id = str(record.get("id", "")).strip()
    work_id = str(record.get("workId", "")).strip()
    text = str(record.get("text", "")).strip()
    previous_text = str(record.get("previousText", "")).strip()
    sentence_type = str(record.get("sentenceType", "")).strip()
    license_id = str(record.get("licenseId", "")).strip().upper()
    license_status = str(record.get("licenseStatus", "approved")).strip().lower()
    source = str(record.get("source", "")).strip()
    annotation_status = str(record.get("annotationStatus", "")).strip()
    chosen_source_value = str(record.get("adjudicationChosenSource", "")).strip()
    adjudication_chosen_source = chosen_source_value or None
    confidence_value = str(record.get("adjudicationConfidence", "")).strip()
    adjudication_confidence = confidence_value or None
    if not example_id:
        raise ValueError(f"id 不能为空: {origin}")
    if not work_id:
        raise ValueError(f"workId 不能为空: {origin}")
    if not text:
        raise ValueError(f"text 不能为空: {origin}")
    if sentence_type not in SENTENCE_TYPES:
        raise ValueError(f"sentenceType 必须是 narration 或 dialogue: {origin}")
    if license_id not in COMMERCIAL_LICENSES:
        raise ValueError(f"licenseId 未列入明确商用白名单: {license_id or '<empty>'}")
    if license_status not in LICENSE_STATUSES:
        raise ValueError(f"licenseStatus 无效: {license_status or '<empty>'}")
    if not source:
        raise ValueError(f"source 不能为空: {origin}")
    if adjudication_chosen_source not in {None, "agent", "qwen", "revised"}:
        raise ValueError(f"adjudicationChosenSource 无效: {adjudication_chosen_source}")
    if adjudication_confidence not in {None, "high", "medium", "low"}:
        raise ValueError(f"adjudicationConfidence 无效: {adjudication_confidence}")

    labels = _emotion_values(record.get("emotions"), "emotions")
    mask_value = record.get("labelMask", {name: 1.0 for name in EMOTION_NAMES})
    label_mask = _emotion_values(mask_value, "labelMask")
    if not any(label_mask):
        raise ValueError(f"labelMask 至少启用一个维度: {origin}")
    intensity = _bounded_number(record.get("intensity", max(labels)), "intensity")
    return EmotionExample(
        example_id=example_id,
        work_id=work_id,
        previous_text=previous_text,
        text=text,
        sentence_type=sentence_type,
        labels=labels,
        label_mask=label_mask,
        intensity=intensity,
        license_id=license_id,
        source=source,
        license_status=license_status,
        annotation_status=annotation_status,
        adjudication_chosen_source=adjudication_chosen_source,
        adjudication_confidence=adjudication_confidence,
    )


def load_jsonl_examples(path: str | Path) -> list[EmotionExample]:
    source_path = Path(path)
    examples: list[EmotionExample] = []
    seen_ids: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败 {source_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"每行必须是 JSON 对象: {source_path}:{line_number}")
            example = parse_example(record, origin=f"{source_path}:{line_number}")
            if example.example_id in seen_ids:
                raise ValueError(f"重复 id={example.example_id}: {source_path}:{line_number}")
            seen_ids.add(example.example_id)
            examples.append(example)
    if not examples:
        raise ValueError(f"数据文件为空: {source_path}")
    return examples


def validate_work_splits(splits: Mapping[str, Iterable[EmotionExample]]) -> None:
    owners: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            previous = owners.setdefault(example.work_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"作品 {example.work_id!r} 同时出现在 {previous!r} 与 {split_name!r}，存在泄漏"
                )
