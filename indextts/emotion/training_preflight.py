"""Read-only preflight for the MacBERT emotion training inputs."""

from __future__ import annotations

import json
import platform
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .annotation import file_sha256
from .context_policy import (
    CONTEXT_MAX_LENGTH,
    CONTEXT_POLICY,
    LEGACY_CONTEXT_DELIMITER,
    LEGACY_CONTEXT_POLICY,
    LEGACY_CONTEXT_SENTENCE_LIMIT,
    ensure_emotion_special_tokens,
    normalize_context_sentence,
    select_target_context,
)
from .dataset import format_current_text
from .schema import EMOTION_NAMES, EmotionExample, load_jsonl_examples, validate_work_splits
from .source_extract import sha256_text


PREFLIGHT_SCHEMA = "readest-emotion-training-preflight-v1"


def validate_resource_requirements(
    *,
    cuda_available: bool,
    free_vram_bytes: int,
    free_disk_bytes: int,
    require_cuda: bool,
    minimum_free_vram_bytes: int,
    minimum_free_disk_bytes: int,
) -> None:
    if require_cuda and not cuda_available:
        raise RuntimeError("训练要求 CUDA，但 PyTorch 无法使用 CUDA")
    if minimum_free_vram_bytes and (
        not cuda_available or free_vram_bytes < minimum_free_vram_bytes
    ):
        raise RuntimeError(
            f"CUDA 空闲显存不足: available={free_vram_bytes / 1024**3:.2f}GiB, "
            f"required={minimum_free_vram_bytes / 1024**3:.2f}GiB"
        )
    if minimum_free_disk_bytes and free_disk_bytes < minimum_free_disk_bytes:
        raise RuntimeError(
            f"训练输出盘空间不足: available={free_disk_bytes / 1024**3:.2f}GiB, "
            f"required={minimum_free_disk_bytes / 1024**3:.2f}GiB"
        )


def _load_many(paths: Sequence[str | Path]) -> tuple[list[EmotionExample], list[dict[str, object]]]:
    examples: list[EmotionExample] = []
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in paths:
        path = Path(source)
        items = load_jsonl_examples(path)
        for item in items:
            if item.example_id in seen:
                raise ValueError(f"跨训练输入重复 id={item.example_id}")
            seen.add(item.example_id)
            examples.append(item)
        files.append({"path": path.as_posix(), "rows": len(items), "sha256": file_sha256(path)})
    if not examples:
        raise ValueError("训练 split 不能为空")
    return examples, files


def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * percentile)
    return int(sorted_values[index])


def _length_summary(lengths: Sequence[int], max_length: int) -> dict[str, object]:
    ordered = sorted(int(value) for value in lengths)
    truncated = sum(value > max_length for value in ordered)
    return {
        "rows": len(ordered),
        "min": ordered[0] if ordered else 0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else 0,
        "overMaxLength": truncated,
        "overMaxLengthRate": truncated / len(ordered) if ordered else 0.0,
    }


def _label_summary(examples: Iterable[EmotionExample]) -> dict[str, object]:
    supervised: Counter[str] = Counter()
    positive: Counter[str] = Counter()
    intensity: Counter[str] = Counter()
    sentence_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    license_statuses: Counter[str] = Counter()
    adjudication_sources: Counter[str] = Counter()
    adjudication_confidences: Counter[str] = Counter()
    count = 0
    for item in examples:
        count += 1
        sentence_types[item.sentence_type] += 1
        sources[item.source] += 1
        licenses[item.license_id] += 1
        license_statuses[item.license_status] += 1
        if item.adjudication_chosen_source:
            adjudication_sources[item.adjudication_chosen_source] += 1
        if item.adjudication_confidence:
            adjudication_confidences[item.adjudication_confidence] += 1
        intensity[f"{item.intensity:g}"] += 1
        for index, name in enumerate(EMOTION_NAMES):
            if item.label_mask[index] > 0.5:
                supervised[name] += 1
                if item.labels[index] > 0.0:
                    positive[name] += 1
    return {
        "rows": count,
        "supervisedExamples": {name: supervised[name] for name in EMOTION_NAMES},
        "positiveExamples": {name: positive[name] for name in EMOTION_NAMES},
        "intensityLevels": dict(sorted(intensity.items())),
        "sentenceTypes": dict(sorted(sentence_types.items())),
        "sources": dict(sorted(sources.items())),
        "licenses": dict(sorted(licenses.items())),
        "licenseStatuses": dict(sorted(license_statuses.items())),
        "adjudicationSources": dict(sorted(adjudication_sources.items())),
        "adjudicationConfidences": dict(sorted(adjudication_confidences.items())),
    }


def _context_leakage(splits: dict[str, Sequence[EmotionExample]]) -> dict[str, object]:
    owners: dict[str, set[str]] = {}
    current_text_owners: dict[str, set[str]] = {}
    for split, examples in splits.items():
        for item in examples:
            if item.context_sentences:
                context_value = json.dumps(
                    [sentence.as_json() for sentence in item.context_sentences],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            else:
                context_value = "\0".join((item.previous_text, item.text, item.sentence_type))
            context_hash = sha256_text(context_value)
            owners.setdefault(context_hash, set()).add(split)
            current_text_owners.setdefault(sha256_text(item.text), set()).add(split)
    return {
        "exactContextHashesAcrossSplits": sum(len(value) > 1 for value in owners.values()),
        "currentTextHashesAcrossSplits": sum(
            len(value) > 1 for value in current_text_owners.values()
        ),
    }


def _token_lengths(
    tokenizer,
    examples: Sequence[EmotionExample],
    *,
    max_length: int,
    batch_size: int = 512,
) -> list[int]:
    lengths: list[int] = []
    del batch_size
    for item in examples:
        if item.context_sentences:
            selection = select_target_context(
                [sentence.as_json() for sentence in item.context_sentences],
                item.target_sentence_id,
                lambda text: len(
                    tokenizer(
                        text,
                        add_special_tokens=True,
                        truncation=False,
                        padding=False,
                        verbose=False,
                    )["input_ids"]
                ),
                max_length=max_length,
            )
            lengths.append(selection.input_token_count)
        else:
            encoded = tokenizer(
                item.previous_text,
                format_current_text(item.text, item.sentence_type),
                add_special_tokens=True,
                truncation=False,
                padding=False,
                verbose=False,
            )
            lengths.append(len(encoded["input_ids"]))
    return lengths


def _complete_context_summary(
    tokenizer,
    examples: Sequence[EmotionExample],
    *,
    max_length: int,
) -> dict[str, object]:
    context_counts: Counter[int] = Counter()
    context_truncation_ids: list[str] = []
    target_truncation_ids: list[str] = []
    invalid_context_ids: list[str] = []
    for item in examples:
        if item.context_sentences:
            records = [sentence.as_json() for sentence in item.context_sentences]
            try:
                selection = select_target_context(
                    records,
                    item.target_sentence_id,
                    lambda text: len(
                        tokenizer(
                            text,
                            add_special_tokens=True,
                            truncation=False,
                            padding=False,
                            verbose=False,
                        )["input_ids"]
                    ),
                    max_length=max_length,
                )
            except ValueError:
                invalid_context_ids.append(item.example_id)
                continue
            if tuple(selection.selected_sentence_ids) != tuple(
                sentence.sentence_id for sentence in item.context_sentences
            ):
                invalid_context_ids.append(item.example_id)
            context_counts[len(item.context_sentences)] += 1
            if selection.target_truncated:
                target_truncation_ids.append(item.example_id)
            if selection.input_token_count > max_length:
                context_truncation_ids.append(item.example_id)
            continue

        raw_sentences = (
            item.previous_text.split(LEGACY_CONTEXT_DELIMITER) if item.previous_text else []
        )
        normalized = [normalize_context_sentence(value) for value in raw_sentences]
        if (
            len(normalized) > LEGACY_CONTEXT_SENTENCE_LIMIT
            or any(not value for value in normalized)
            or LEGACY_CONTEXT_DELIMITER.join(normalized) != item.previous_text
        ):
            invalid_context_ids.append(item.example_id)
        context_counts[len(normalized)] += 1
        current = format_current_text(item.text, item.sentence_type)
        target_length = len(
            tokenizer(
                "",
                current,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                verbose=False,
            )["input_ids"]
        )
        pair_length = len(
            tokenizer(
                item.previous_text,
                current,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                verbose=False,
            )["input_ids"]
        )
        if target_length > max_length:
            target_truncation_ids.append(item.example_id)
            if item.previous_text:
                invalid_context_ids.append(item.example_id)
        elif pair_length > max_length:
            context_truncation_ids.append(item.example_id)
    return {
        "policy": (
            CONTEXT_POLICY
            if any(item.context_sentences for item in examples)
            else LEGACY_CONTEXT_POLICY
        ),
        "contextSentenceCounts": {
            str(count): rows for count, rows in sorted(context_counts.items())
        },
        "targetTruncationCount": len(target_truncation_ids),
        "targetTruncationIds": target_truncation_ids,
        "contextTruncationCount": len(context_truncation_ids),
        "contextTruncationIds": context_truncation_ids,
        "invalidContextCount": len(set(invalid_context_ids)),
        "invalidContextIds": sorted(set(invalid_context_ids)),
    }


def preflight_training(
    train_paths: Sequence[str | Path],
    dev_paths: Sequence[str | Path],
    test_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    base_model: str | Path = "hfl/chinese-macbert-base",
    max_length: int = CONTEXT_MAX_LENGTH,
    verify_base_weights: bool = False,
    minimum_positive: int = 1_000,
    require_cuda: bool = False,
    minimum_free_vram_bytes: int = 0,
    minimum_free_disk_bytes: int = 0,
    require_complete_context: bool = False,
) -> dict[str, object]:
    """Validate local inputs and model cache without running optimization steps."""

    if max_length <= 0:
        raise ValueError("max_length 必须为正数")
    if minimum_positive < 0:
        raise ValueError("minimum_positive 不能为负数")
    if minimum_free_vram_bytes < 0 or minimum_free_disk_bytes < 0:
        raise ValueError("最低显存和磁盘空间不能为负数")
    splits = {}
    input_files = {}
    for split, paths in (("train", train_paths), ("dev", dev_paths), ("test", test_paths)):
        examples, files = _load_many(paths)
        splits[split] = examples
        input_files[split] = files
    validate_work_splits(splits)
    leakage = _context_leakage(splits)
    if leakage["exactContextHashesAcrossSplits"]:
        raise ValueError("训练输入存在跨 split 的完整上下文泄漏")

    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("训练预检需要主环境中的 torch 与 transformers") from exc

    model_name = str(base_model)
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True, use_fast=True
    )
    if any(item.context_sentences for examples in splits.values() for item in examples):
        ensure_emotion_special_tokens(tokenizer)
    max_position_embeddings = int(getattr(config, "max_position_embeddings", 0))
    if max_position_embeddings and max_length > max_position_embeddings:
        raise ValueError(
            f"max_length={max_length} 超过基础模型位置上限 {max_position_embeddings}"
        )
    weight_report: dict[str, object] = {"verified": False}
    if verify_base_weights:
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        dtype_counts: Counter[str] = Counter()
        parameter_count = 0
        for parameter in model.parameters():
            parameter_count += parameter.numel()
            dtype_counts[str(parameter.dtype).replace("torch.", "")] += parameter.numel()
        weight_report = {
            "verified": True,
            "parameterCount": parameter_count,
            "parameterDtypes": dict(sorted(dtype_counts.items())),
        }
        del model

    length_report = {
        split: _length_summary(
            _token_lengths(tokenizer, examples, max_length=max_length), max_length
        )
        for split, examples in splits.items()
    }
    complete_context = {
        split: _complete_context_summary(tokenizer, examples, max_length=max_length)
        for split, examples in splits.items()
    }
    if require_complete_context:
        failures = {
            split: summary
            for split, summary in complete_context.items()
            if summary["contextTruncationCount"] or summary["invalidContextCount"]
        }
        if failures:
            compact = {
                split: {
                    "contextTruncationCount": summary["contextTruncationCount"],
                    "invalidContextCount": summary["invalidContextCount"],
                }
                for split, summary in failures.items()
            }
            raise ValueError(f"完整上文策略校验失败: {compact}")
    warnings: list[str] = []
    for split, summary in length_report.items():
        rate = float(summary["overMaxLengthRate"])
        if rate > 0.05:
            warnings.append(
                f"{split} 有 {rate:.2%} 的样本超过 maxLength={max_length}"
            )

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "deviceCount": int(torch.cuda.device_count()),
        "devices": [],
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "totalMemoryBytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        free_memory, visible_total = torch.cuda.mem_get_info(0)
        cuda["selectedDevice"] = 0
        cuda["freeMemoryBytes"] = int(free_memory)
        cuda["visibleTotalMemoryBytes"] = int(visible_total)
    target = Path(output_path)
    disk_probe = target.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    usage = shutil.disk_usage(disk_probe)
    disk = {
        "path": disk_probe.resolve().as_posix(),
        "totalBytes": int(usage.total),
        "usedBytes": int(usage.used),
        "freeBytes": int(usage.free),
    }
    validate_resource_requirements(
        cuda_available=bool(torch.cuda.is_available()),
        free_vram_bytes=int(cuda.get("freeMemoryBytes", 0)),
        free_disk_bytes=int(usage.free),
        require_cuda=require_cuda,
        minimum_free_vram_bytes=minimum_free_vram_bytes,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
    )

    label_reports = {name: _label_summary(items) for name, items in splits.items()}
    training_positive = label_reports["train"]["positiveExamples"]
    assert isinstance(training_positive, dict)
    positive_gaps = {
        name: max(0, minimum_positive - int(training_positive[name]))
        for name in EMOTION_NAMES
    }
    from .imbalance import dimension_weights

    balanced_weights, balanced_weight_report = dimension_weights(splits["train"])
    if any(positive_gaps.values()):
        warnings.append(
            "训练集仍有情感正例覆盖缺口: "
            + ", ".join(
                f"{name}={gap}" for name, gap in positive_gaps.items() if gap
            )
        )

    report: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "trainingStarted": False,
        "preflightPassed": True,
        "baseModel": model_name,
        "baseModelCache": {
            "configVerified": True,
            "tokenizerVerified": True,
            "weights": weight_report,
            "modelType": str(getattr(config, "model_type", "")),
            "hiddenSize": int(getattr(config, "hidden_size", 0)),
            "numHiddenLayers": int(getattr(config, "num_hidden_layers", 0)),
            "maxPositionEmbeddings": max_position_embeddings,
            "vocabSize": int(getattr(config, "vocab_size", 0)),
        },
        "maxLength": max_length,
        "inputs": input_files,
        "splitCounts": {name: len(items) for name, items in splits.items()},
        "splitWorks": {
            name: len({item.work_id for item in items}) for name, items in splits.items()
        },
        "labels": label_reports,
        "dataReadiness": {
            "minimumPositivePerEmotion": minimum_positive,
            "qualityCoverageReady": not any(positive_gaps.values()),
            "remainingPositiveGaps": positive_gaps,
            "dimensionWeights": {
                name: float(value)
                for name, value in zip(EMOTION_NAMES, balanced_weights.tolist())
            },
            "dimensionWeightReport": balanced_weight_report,
        },
        "tokenLengths": length_report,
        "completeContext": complete_context,
        "completeContextRequired": require_complete_context,
        "crossSplitLeakage": leakage,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": cuda,
            "disk": disk,
        },
        "warnings": warnings,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
