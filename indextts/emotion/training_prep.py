"""Build a deterministic, test-only training snapshot after adjudication.

The original finalized dataset and adjudication artifacts are immutable inputs.
This module applies second decisions to a new snapshot, quarantines low-confidence
decisions, and records enough hashes and counts to reproduce the exact inputs to
the MacBERT trainer without starting a training job.
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .annotation import ALLOWED_LEVELS, AnnotationValidationError, file_sha256
from .schema import EMOTION_NAMES, load_jsonl_examples, validate_work_splits
from .source_extract import sha256_text


TRAINING_PREP_SCHEMA = "readest-emotion-training-prep-v1"
ADJUDICATION_RESULT_SCHEMA = "readest-emotion-adjudication-result-v1"
ADJUDICATION_DECISION_SCHEMA = "readest-emotion-adjudication-v1"
ALLOWED_PRIMARY = frozenset((*EMOTION_NAMES, "mixed", "base"))
ALLOWED_SOURCES = frozenset({"agent", "qwen", "revised"})
ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(
                    f"JSONL 解析失败 {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise AnnotationValidationError(
                    f"JSONL 第 {line_number} 行不是对象: {source}"
                )
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _emotion_vector(value: object, *, source: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(EMOTION_NAMES):
        raise AnnotationValidationError(f"{source} 的情感维度不完整")
    result: dict[str, float] = {}
    for name in EMOTION_NAMES:
        try:
            number = float(value[name])
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError(f"{source}.{name} 必须是数值") from exc
        if number not in ALLOWED_LEVELS:
            raise AnnotationValidationError(
                f"{source}.{name} 必须是 0/0.33/0.67/1"
            )
        result[name] = number
    return result


def _level(value: object, *, source: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError(f"{source} 必须是数值") from exc
    if number not in ALLOWED_LEVELS:
        raise AnnotationValidationError(f"{source} 必须是 0/0.33/0.67/1")
    return number


def _same_vector(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(float(left[name]) == float(right[name]) for name in EMOTION_NAMES)


def _validate_adjudication(
    row: Mapping[str, object],
    base: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    if row.get("schema") != ADJUDICATION_RESULT_SCHEMA:
        raise AnnotationValidationError(f"{source} 的结果 schema 无效")
    candidate_id = str(base.get("id", ""))
    if str(row.get("id", "")) != candidate_id:
        raise AnnotationValidationError(f"{source} 的 id 与训练基线不一致")
    for field in ("workId", "sentenceType", "previousText", "text"):
        if str(row.get(field, "")) != str(base.get(field, "")):
            raise AnnotationValidationError(
                f"{source} 的 {field} 与训练基线不一致: {candidate_id}"
            )

    agent = row.get("agentJudgment")
    qwen = row.get("qwenJudgment")
    decision = row.get("secondDecision")
    if not isinstance(agent, Mapping) or not isinstance(qwen, Mapping):
        raise AnnotationValidationError(f"{source} 缺少 agent/Qwen 两套判定")
    if not isinstance(decision, Mapping):
        raise AnnotationValidationError(f"{source} 缺少 secondDecision")
    if decision.get("schema") != ADJUDICATION_DECISION_SCHEMA:
        raise AnnotationValidationError(f"{source} 的 secondDecision schema 无效")
    if str(decision.get("id", "")) != candidate_id:
        raise AnnotationValidationError(f"{source} 的 secondDecision id 不一致")

    expected_hashes = {
        "textSha256": sha256_text(str(base.get("text", ""))),
        "previousTextSha256": sha256_text(str(base.get("previousText", ""))),
    }
    for field, expected in expected_hashes.items():
        if str(decision.get(field, "")) != expected:
            raise AnnotationValidationError(f"{source} 的 {field} 与训练基线不一致")

    agent_emotions = _emotion_vector(
        agent.get("emotions"), source=f"{source}.agentJudgment.emotions"
    )
    qwen_emotions = _emotion_vector(
        qwen.get("emotionsNeutralAdjusted"),
        source=f"{source}.qwenJudgment.emotionsNeutralAdjusted",
    )
    base_emotions = _emotion_vector(
        base.get("emotions"), source=f"{source}.baseline.emotions"
    )
    if not _same_vector(agent_emotions, base_emotions):
        raise AnnotationValidationError(
            f"{source} 的 agent 判定与当前训练基线不一致: {candidate_id}"
        )
    agent_intensity = _level(
        agent.get("intensity"), source=f"{source}.agentJudgment.intensity"
    )
    base_intensity = _level(
        base.get("intensity"), source=f"{source}.baseline.intensity"
    )
    if agent_intensity != base_intensity:
        raise AnnotationValidationError(
            f"{source} 的 agent 强度与当前训练基线不一致: {candidate_id}"
        )

    maximum_difference = max(
        abs(agent_emotions[name] - qwen_emotions[name]) for name in EMOTION_NAMES
    )
    if maximum_difference < 0.67:
        raise AnnotationValidationError(
            f"{source} 不属于大分歧池: maxDifference={maximum_difference:g}"
        )

    disposition = str(decision.get("disposition", ""))
    if disposition not in {"keep", "drop"}:
        raise AnnotationValidationError(f"{source} 的 disposition 无效")
    chosen_source = str(decision.get("chosenSource", ""))
    if chosen_source not in ALLOWED_SOURCES:
        raise AnnotationValidationError(f"{source} 的 chosenSource 无效")
    confidence = str(decision.get("confidence", ""))
    if confidence not in ALLOWED_CONFIDENCE:
        raise AnnotationValidationError(f"{source} 的 confidence 无效")
    final_emotions = _emotion_vector(
        decision.get("emotions"), source=f"{source}.secondDecision.emotions"
    )
    intensity = _level(
        decision.get("intensity"), source=f"{source}.secondDecision.intensity"
    )
    primary = str(decision.get("primaryEmotion", ""))
    if primary not in ALLOWED_PRIMARY:
        raise AnnotationValidationError(f"{source} 的 primaryEmotion 无效")
    is_base = not any(final_emotions.values())
    if is_base != (primary == "base") or is_base != (intensity == 0.0):
        raise AnnotationValidationError(
            f"{source} 的 base、primaryEmotion 与 intensity 不一致"
        )
    return {
        "id": candidate_id,
        "disposition": disposition,
        "chosenSource": chosen_source,
        "confidence": confidence,
        "primaryEmotion": primary,
        "emotions": final_emotions,
        "intensity": intensity,
        "rationale": str(decision.get("rationale", "")).strip(),
        "maximumDifference": maximum_difference,
        "agentEmotions": agent_emotions,
        "qwenEmotionsNeutralAdjusted": qwen_emotions,
    }


def _dataset_statistics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    positives: Counter[str] = Counter()
    sentence_types: Counter[str] = Counter()
    intensity_levels: Counter[str] = Counter()
    neutral_count = 0
    for row in rows:
        emotions = row["emotions"]
        assert isinstance(emotions, Mapping)
        sentence_types[str(row["sentenceType"])] += 1
        intensity_levels[f"{float(row['intensity']):g}"] += 1
        if not any(float(emotions[name]) for name in EMOTION_NAMES):
            neutral_count += 1
        for name in EMOTION_NAMES:
            if float(emotions[name]) > 0.0:
                positives[name] += 1
    return {
        "rows": len(rows),
        "neutralBaseExamples": neutral_count,
        "activeExamples": len(rows) - neutral_count,
        "positiveExamples": {name: positives[name] for name in EMOTION_NAMES},
        "sentenceTypes": dict(sorted(sentence_types.items())),
        "intensityLevels": dict(sorted(intensity_levels.items())),
    }


def _cross_split_leakage(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    context_owners: dict[str, set[str]] = {}
    text_owners: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            text = str(row["text"])
            context = "\0".join(
                (str(row.get("previousText", "")), text, str(row["sentenceType"]))
            )
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            context_owners.setdefault(context_hash, set()).add(split)
            text_owners.setdefault(text_hash, set()).add(split)
    return {
        "exactContextHashesAcrossSplits": sum(
            len(owners) > 1 for owners in context_owners.values()
        ),
        "currentTextHashesAcrossSplits": sum(
            len(owners) > 1 for owners in text_owners.values()
        ),
        "exactContextLeakage": any(len(owners) > 1 for owners in context_owners.values()),
    }


def _verify_brighter(brighter_dir: str | Path | None) -> dict[str, object] | None:
    if brighter_dir is None:
        return None
    root = Path(brighter_dir)
    result: dict[str, object] = {"directory": root.as_posix(), "splits": {}}
    split_examples = {}
    for split in ("train", "dev", "test"):
        path = root / f"brighter-chn-{split}.jsonl"
        if not path.is_file():
            raise AnnotationValidationError(f"缺少本地 BRIGHTER 数据: {path}")
        examples = load_jsonl_examples(path)
        split_examples[split] = examples
        supervised: Counter[str] = Counter()
        positives: Counter[str] = Counter()
        for example in examples:
            for index, name in enumerate(EMOTION_NAMES):
                if example.label_mask[index] > 0.5:
                    supervised[name] += 1
                    if example.labels[index] > 0.0:
                        positives[name] += 1
        result["splits"][split] = {
            "rows": len(examples),
            "sha256": file_sha256(path),
            "maskedDimensions": ["melancholic", "calm"],
            "supervisedExamples": {
                name: supervised[name] for name in EMOTION_NAMES
            },
            "positiveExamples": {name: positives[name] for name in EMOTION_NAMES},
        }
    validate_work_splits(split_examples)
    attribution = root / "BRIGHTER-ATTRIBUTION.json"
    if not attribution.is_file():
        raise AnnotationValidationError(f"缺少 BRIGHTER 署名文件: {attribution}")
    result["attributionSha256"] = file_sha256(attribution)
    return result


def prepare_training_data(
    base_dir: str | Path,
    adjudication_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    brighter_dir: str | Path | None = None,
    expected_adjudications: int | None = None,
    base_model: str = "hfl/chinese-macbert-base",
    seed: int = 20260829,
    max_length: int = 256,
) -> dict[str, object]:
    """Apply adjudications and materialize a training snapshot without training."""

    if not adjudication_paths:
        raise ValueError("至少需要一个二次决断文件")
    base_root = Path(base_dir)
    target = Path(output_dir)
    if target.resolve() == base_root.resolve():
        raise ValueError("输出目录不能覆盖原始 final-test 数据")

    split_rows: dict[str, list[dict[str, object]]] = {}
    base_by_id: dict[str, dict[str, object]] = {}
    split_by_id: dict[str, str] = {}
    base_hashes: dict[str, str] = {}
    parsed_splits = {}
    for split in ("train", "dev", "test"):
        path = base_root / f"{split}.jsonl"
        if not path.is_file():
            raise AnnotationValidationError(f"缺少训练基线 split: {path}")
        rows = _read_jsonl(path)
        split_rows[split] = rows
        parsed_splits[split] = load_jsonl_examples(path)
        base_hashes[split] = file_sha256(path)
        for row in rows:
            candidate_id = str(row.get("id", ""))
            if not candidate_id or candidate_id in base_by_id:
                raise AnnotationValidationError(f"训练基线含空或重复 id: {candidate_id}")
            base_by_id[candidate_id] = row
            split_by_id[candidate_id] = split
    validate_work_splits(parsed_splits)

    decisions: dict[str, dict[str, object]] = {}
    adjudication_inputs: list[dict[str, object]] = []
    for adjudication_path in adjudication_paths:
        path = Path(adjudication_path)
        rows = _read_jsonl(path)
        adjudication_inputs.append(
            {"path": path.as_posix(), "rows": len(rows), "sha256": file_sha256(path)}
        )
        for line_number, row in enumerate(rows, 1):
            candidate_id = str(row.get("id", ""))
            if candidate_id in decisions:
                raise AnnotationValidationError(f"二次决断重复 id: {candidate_id}")
            base = base_by_id.get(candidate_id)
            if base is None:
                raise AnnotationValidationError(
                    f"二次决断 id 不存在于训练基线: {candidate_id}"
                )
            decisions[candidate_id] = _validate_adjudication(
                row, base, source=f"{path}:{line_number}"
            )
    if expected_adjudications is not None and len(decisions) != expected_adjudications:
        raise AnnotationValidationError(
            f"二次决断数量不符: actual={len(decisions)}, expected={expected_adjudications}"
        )

    target.mkdir(parents=True, exist_ok=True)
    output_splits: dict[str, list[dict[str, object]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    audit_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    chosen_sources: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    adjudicated_by_split: Counter[str] = Counter()
    changed_by_split: Counter[str] = Counter()
    dropped_by_split: Counter[str] = Counter()
    quarantined_by_split: Counter[str] = Counter()

    for split, rows in split_rows.items():
        for base in rows:
            candidate_id = str(base["id"])
            decision = decisions.get(candidate_id)
            if decision is None:
                output_splits[split].append(dict(base))
                continue
            adjudicated_by_split[split] += 1
            chosen_sources[str(decision["chosenSource"])] += 1
            confidence_counts[str(decision["confidence"])] += 1
            old_emotions = _emotion_vector(
                base.get("emotions"), source=f"baseline.{candidate_id}.emotions"
            )
            old_intensity = float(base["intensity"])
            final_emotions = decision["emotions"]
            assert isinstance(final_emotions, dict)
            changed = not _same_vector(old_emotions, final_emotions) or (
                old_intensity != float(decision["intensity"])
            )
            changed_by_split[split] += int(changed)
            audit = {
                "schema": TRAINING_PREP_SCHEMA,
                "id": candidate_id,
                "split": split,
                "workId": str(base["workId"]),
                "chosenSource": decision["chosenSource"],
                "confidence": decision["confidence"],
                "primaryEmotion": decision["primaryEmotion"],
                "maximumDifference": decision["maximumDifference"],
                "changed": changed,
                "originalEmotions": old_emotions,
                "originalIntensity": old_intensity,
                "finalEmotions": final_emotions,
                "finalIntensity": decision["intensity"],
            }
            audit_rows.append(audit)
            if decision["disposition"] == "drop":
                dropped_by_split[split] += 1
                continue
            updated = dict(base)
            updated["emotions"] = final_emotions
            updated["intensity"] = decision["intensity"]
            updated["annotationStatus"] = "second-agent-adjudicated"
            updated["adjudicationChosenSource"] = decision["chosenSource"]
            updated["adjudicationConfidence"] = decision["confidence"]
            updated["adjudicationPrimaryEmotion"] = decision["primaryEmotion"]
            if decision["confidence"] == "low":
                quarantined_by_split[split] += 1
                quarantine_rows.append(
                    {
                        **audit,
                        "previousText": str(base.get("previousText", "")),
                        "text": str(base["text"]),
                        "sentenceType": str(base["sentenceType"]),
                        "rationale": decision["rationale"],
                        "agentEmotions": decision["agentEmotions"],
                        "qwenEmotionsNeutralAdjusted": decision[
                            "qwenEmotionsNeutralAdjusted"
                        ],
                    }
                )
                continue
            output_splits[split].append(updated)

    output_hashes: dict[str, str] = {}
    for split, rows in output_splits.items():
        path = target / f"{split}.jsonl"
        _write_jsonl(path, rows)
        output_hashes[split] = file_sha256(path)
    audit_path = target / "adjudication-audit.jsonl"
    quarantine_path = target / "review-required-low-confidence.jsonl"
    _write_jsonl(audit_path, audit_rows)
    _write_jsonl(quarantine_path, quarantine_rows)

    parsed_outputs = {
        split: load_jsonl_examples(target / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    validate_work_splits(parsed_outputs)
    if sum(len(items) for items in output_splits.values()) + len(quarantine_rows) + sum(
        dropped_by_split.values()
    ) != len(base_by_id):
        raise AnnotationValidationError("输出、隔离与 drop 未完整覆盖训练基线")

    brighter = _verify_brighter(brighter_dir)
    all_output_rows = [row for rows in output_splits.values() for row in rows]
    split_statistics = {
        split: _dataset_statistics(rows) for split, rows in output_splits.items()
    }
    combined_training_positives = dict(
        split_statistics["train"]["positiveExamples"]
    )
    if brighter is not None:
        brighter_splits = brighter.get("splits")
        if isinstance(brighter_splits, Mapping):
            brighter_train = brighter_splits.get("train")
            if isinstance(brighter_train, Mapping):
                brighter_positives = brighter_train.get("positiveExamples")
                if isinstance(brighter_positives, Mapping):
                    for name in EMOTION_NAMES:
                        combined_training_positives[name] += int(
                            brighter_positives.get(name, 0)
                        )
    target_positive_per_emotion = 1000
    coverage_gaps = {
        name: max(0, target_positive_per_emotion - combined_training_positives[name])
        for name in EMOTION_NAMES
        if combined_training_positives[name] < target_positive_per_emotion
    }
    leakage = _cross_split_leakage(output_splits)
    if leakage["exactContextLeakage"]:
        raise AnnotationValidationError("训练快照存在跨 split 的完整上下文泄漏")
    formal_blockers = [
        "dataset-test-only",
        "independent-agent-adjudication-not-human-signoff",
    ]
    if quarantine_rows:
        formal_blockers.append(f"low-confidence-adjudications:{len(quarantine_rows)}")
    if chosen_sources["qwen"]:
        formal_blockers.append(f"qwen-informed-second-decisions:{chosen_sources['qwen']}")
    if coverage_gaps:
        formal_blockers.append(
            "emotion-positive-coverage-gaps:"
            + ",".join(f"{name}={gap}" for name, gap in coverage_gaps.items())
        )
    report: dict[str, object] = {
        "schema": TRAINING_PREP_SCHEMA,
        "datasetStatus": "candidate-training-prepared",
        "licenseStatus": "test-only",
        "trainingStarted": False,
        "testTrainingReady": True,
        "formalTrainingReady": False,
        "formalTrainingBlockers": formal_blockers,
        "baseModel": base_model,
        "precision": "fp32",
        "seed": seed,
        "maxLength": max_length,
        "baseDataset": {
            "directory": base_root.as_posix(),
            "rows": len(base_by_id),
            "splitSha256": base_hashes,
        },
        "adjudicationInputs": adjudication_inputs,
        "adjudicationRows": len(decisions),
        "adjudicatedBySplit": dict(sorted(adjudicated_by_split.items())),
        "changedRows": sum(changed_by_split.values()),
        "changedBySplit": dict(sorted(changed_by_split.items())),
        "chosenSource": {name: chosen_sources[name] for name in sorted(ALLOWED_SOURCES)},
        "confidence": {name: confidence_counts[name] for name in sorted(ALLOWED_CONFIDENCE)},
        "droppedBySplit": dict(sorted(dropped_by_split.items())),
        "quarantinedLowConfidence": len(quarantine_rows),
        "quarantinedBySplit": dict(sorted(quarantined_by_split.items())),
        "outputSplitCounts": {name: len(rows) for name, rows in output_splits.items()},
        "outputSplitWorks": {
            name: len({str(row["workId"]) for row in rows})
            for name, rows in output_splits.items()
        },
        "statistics": _dataset_statistics(all_output_rows),
        "splitStatistics": split_statistics,
        "crossSplitLeakage": leakage,
        "trainingCoverage": {
            "targetPositiveExamplesPerEmotion": target_positive_per_emotion,
            "combinedNovelAndBrighterTrainPositiveExamples": combined_training_positives,
            "gaps": coverage_gaps,
            "meetsTarget": not coverage_gaps,
        },
        "brighter": brighter,
        "outputs": {
            "splits": output_hashes,
            "adjudicationAudit": {
                "rows": len(audit_rows),
                "sha256": file_sha256(audit_path),
            },
            "lowConfidenceReview": {
                "rows": len(quarantine_rows),
                "sha256": file_sha256(quarantine_path),
            },
        },
        "plannedTrainingInputs": {
            "train": [
                (target / "train.jsonl").as_posix(),
                (Path(brighter_dir) / "brighter-chn-train.jsonl").as_posix()
                if brighter_dir
                else None,
            ],
            "dev": [
                (target / "dev.jsonl").as_posix(),
                (Path(brighter_dir) / "brighter-chn-dev.jsonl").as_posix()
                if brighter_dir
                else None,
            ],
            "test": [
                (target / "test.jsonl").as_posix(),
                (Path(brighter_dir) / "brighter-chn-test.jsonl").as_posix()
                if brighter_dir
                else None,
            ],
        },
    }
    for values in report["plannedTrainingInputs"].values():
        assert isinstance(values, list)
        values[:] = [value for value in values if value is not None]
    report_path = target / "training-prep-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
