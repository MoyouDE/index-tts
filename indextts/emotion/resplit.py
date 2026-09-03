"""Deterministic work-level re-splitting for emotion training snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .schema import EMOTION_NAMES, EmotionExample, load_jsonl_examples, validate_work_splits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, examples: Iterable[EmotionExample]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_json(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _split_statistics(examples: Sequence[EmotionExample]) -> dict[str, object]:
    positive = {
        name: sum(
            example.label_mask[index] > 0.5 and example.labels[index] >= 0.35
            for example in examples
        )
        for index, name in enumerate(EMOTION_NAMES)
    }
    observed = {
        name: sum(example.label_mask[index] > 0.5 for example in examples)
        for index, name in enumerate(EMOTION_NAMES)
    }
    return {
        "examples": len(examples),
        "works": sorted({example.work_id for example in examples}),
        "sentenceTypes": dict(sorted(Counter(item.sentence_type for item in examples).items())),
        "baseExamples": sum(example.intensity <= 0.05 for example in examples),
        "positiveExamples": positive,
        "observedExamples": observed,
    }


def resplit_by_work(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    dev_works: Sequence[str],
    test_works: Sequence[str],
    minimum_dev_positive: int = 1,
    minimum_test_positive: int = 1,
) -> dict[str, object]:
    """Merge snapshots, then split exclusively by stable work identifiers."""

    if not input_paths:
        raise ValueError("至少需要一个输入数据文件")
    if minimum_dev_positive < 0 or minimum_test_positive < 0:
        raise ValueError("最小正例数不能为负数")
    dev_set = {item.strip() for item in dev_works if item.strip()}
    test_set = {item.strip() for item in test_works if item.strip()}
    if not dev_set or not test_set:
        raise ValueError("dev_works 和 test_works 均不能为空")
    overlap = dev_set & test_set
    if overlap:
        raise ValueError(f"dev/test 作品重复: {sorted(overlap)}")

    paths = [Path(path) for path in input_paths]
    examples: list[EmotionExample] = []
    seen_ids: set[str] = set()
    for path in paths:
        for example in load_jsonl_examples(path):
            if example.example_id in seen_ids:
                raise ValueError(f"跨输入文件重复 id={example.example_id}")
            seen_ids.add(example.example_id)
            examples.append(example)
    available_works = {example.work_id for example in examples}
    missing = (dev_set | test_set) - available_works
    if missing:
        raise ValueError(f"指定作品不存在: {sorted(missing)}")

    splits: dict[str, list[EmotionExample]] = {"train": [], "dev": [], "test": []}
    for example in examples:
        split = "dev" if example.work_id in dev_set else "test" if example.work_id in test_set else "train"
        splits[split].append(example)
    for split_examples in splits.values():
        split_examples.sort(key=lambda item: (item.work_id, item.example_id))
    if not splits["train"]:
        raise ValueError("训练集不能为空")
    validate_work_splits(splits)

    statistics = {name: _split_statistics(items) for name, items in splits.items()}
    for name, minimum in (("dev", minimum_dev_positive), ("test", minimum_test_positive)):
        positive = statistics[name]["positiveExamples"]
        insufficient = {
            emotion: count for emotion, count in positive.items() if int(count) < minimum
        }
        if insufficient:
            raise ValueError(f"{name} 情感正例覆盖不足（要求 {minimum}）: {insufficient}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        _write_jsonl(target / f"{name}.jsonl", items)
    report = {
        "schema": "readest-emotion-work-resplit-v1",
        "policy": "work-disjoint; label-coverage-balanced",
        "inputs": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "minimumPositive": {
            "dev": minimum_dev_positive,
            "test": minimum_test_positive,
        },
        "splits": statistics,
        "outputs": {
            name: {
                "path": str(target / f"{name}.jsonl"),
                "sha256": _sha256(target / f"{name}.jsonl"),
            }
            for name in splits
        },
    }
    report_path = target / "resplit-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
