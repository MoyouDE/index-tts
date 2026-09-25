"""Frozen, descriptive evaluation for a versioned one-click emotion run.

The only selection threshold is calibrated on dev. The existing test split is
evaluated for the final report, never used to change training configuration.
Nothing in this module promotes or publishes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indextts.emotion.imbalance_metrics import detailed_metrics
from indextts.emotion.metrics import calibrate_neutral_threshold
from indextts.emotion.model import load_checkpoint
from indextts.emotion.schema import EMOTION_NAMES, load_jsonl_examples
from indextts.emotion.train import collect_model_predictions


SCHEMA = "readest-emotion-one-click-evaluation-v1"
DEV_SAFETY = {
    "maxMacroF1Drop": 0.01,
    "maxMacroSpearmanDrop": 0.01,
    "maxZeroFarIncrease": 0.01,
    "maxAuxiliaryMaeIncrease": 0.01,
}
ADDED_TEST_GATES = {
    "minMacroF1": 0.48,
    "minMacroSpearman": 0.45,
    "maxNeutralFar": 0.5,
    "maxIntensityMae": 0.2,
    "minNarrationMacroF1": 0.45,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_rows(predictions: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in predictions.items()}


def slice_metrics(
    predictions: dict[str, np.ndarray], rows: list, neutral_threshold: float,
    added_prefix: str,
) -> dict[str, dict]:
    ids = np.array([row.example_id.startswith(added_prefix) for row in rows], dtype=bool)
    narration = np.array([row.sentence_type == "narration" for row in rows], dtype=bool)
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "old": ~ids,
        "added": ids,
        "addedDialogue": ids & ~narration,
        "addedNarration": ids & narration,
    }
    return {
        name: detailed_metrics(select_rows(predictions, mask), neutral_threshold)
        for name, mask in masks.items() if mask.any()
    }


def quality_checks(formal_dev: dict, candidate_dev: dict, candidate_test: dict) -> dict[str, bool]:
    old, new = formal_dev, candidate_dev
    added = candidate_test["added"]
    narration = candidate_test["addedNarration"]
    return {
        "oldDevMacroF1": new["macroF1"] >= old["macroF1"] - DEV_SAFETY["maxMacroF1Drop"],
        "oldDevMacroSpearman": new["macroSpearman"] >= old["macroSpearman"] - DEV_SAFETY["maxMacroSpearmanDrop"],
        "oldDevZeroFar": new["zeroDimensionFalseActivationRate"] <= old["zeroDimensionFalseActivationRate"] + DEV_SAFETY["maxZeroFarIncrease"],
        "oldDevAuxiliaryMae": new["auxiliary"]["mae"] <= old["auxiliary"]["mae"] + DEV_SAFETY["maxAuxiliaryMaeIncrease"],
        "addedTestMacroF1": added["macroF1"] >= ADDED_TEST_GATES["minMacroF1"],
        "addedTestMacroSpearman": added["macroSpearman"] >= ADDED_TEST_GATES["minMacroSpearman"],
        "addedTestNeutralFar": added["neutralFalseActivationRate"] <= ADDED_TEST_GATES["maxNeutralFar"],
        "addedTestIntensityMae": added["intensityMae"] <= ADDED_TEST_GATES["maxIntensityMae"],
        "addedNarrationMacroF1": narration["macroF1"] >= ADDED_TEST_GATES["minNarrationMacroF1"],
    }


def load_or_predict(
    model, tokenizer, checkpoint: Path, source: Path, rows: list,
    destination: Path, *, device: torch.device, batch_size: int, max_length: int,
) -> dict[str, np.ndarray]:
    weights = checkpoint / "model.safetensors"
    identity = {
        "modelSha256": file_sha256(weights),
        "inputSha256": file_sha256(source),
        "sampleCount": len(rows),
        "batchMaxLength": max_length,
    }
    archive_path = destination.with_suffix(".npz")
    metadata_path = destination.with_suffix(".meta.json")
    if archive_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity or metadata.get("sha256") != file_sha256(archive_path):
            raise ValueError(f"预测缓存指纹不匹配，请检查输出目录: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    predictions = collect_model_predictions(
        model, rows, tokenizer, device=device, batch_size=batch_size,
        max_length=max_length, progress=True,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **predictions)
    os.replace(temporary, archive_path)
    write_json(metadata_path, {"identity": identity, "sha256": file_sha256(archive_path)})
    return predictions


def write_prediction_rows(path: Path, rows: list, predictions: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            record = {
                "id": row.example_id,
                "workId": row.work_id,
                "sentenceType": row.sentence_type,
                "labels": dict(zip(EMOTION_NAMES, map(float, predictions["labels"][index]))),
                "labelMask": dict(zip(EMOTION_NAMES, map(float, predictions["labelMasks"][index]))),
                "predictions": dict(zip(EMOTION_NAMES, map(float, predictions["vectors"][index]))),
                "intensityLabel": float(np.asarray(predictions["intensities"][index]).item()),
                "predictedIntensity": float(np.asarray(predictions["predictedIntensities"][index]).item()),
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def added_test_cases(
    rows: list, formal: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
    added_prefix: str, limit: int = 10,
) -> dict[str, list[dict]]:
    cases = []
    for index, row in enumerate(rows):
        if not row.example_id.startswith(added_prefix):
            continue
        labels = formal["labels"][index]
        valid = formal["labelMasks"][index] > 0.5
        old_mae = float(np.abs(formal["vectors"][index][valid] - labels[valid]).mean())
        new_mae = float(np.abs(candidate["vectors"][index][valid] - labels[valid]).mean())
        cases.append({
            "id": row.example_id, "workId": row.work_id,
            "sentenceType": row.sentence_type,
            "formalMae": old_mae, "candidateMae": new_mae,
            "deltaMae": new_mae - old_mae,
            "labels": dict(zip(EMOTION_NAMES, map(float, labels))),
            "formal": dict(zip(EMOTION_NAMES, map(float, formal["vectors"][index]))),
            "candidate": dict(zip(EMOTION_NAMES, map(float, candidate["vectors"][index]))),
        })
    return {
        "improved": sorted(cases, key=lambda row: (row["deltaMae"], row["id"]))[:limit],
        "degraded": sorted(cases, key=lambda row: (-row["deltaMae"], row["id"]))[:limit],
    }


def render_report(report: dict) -> str:
    def fmt(value: float) -> str:
        return f"{value:.4f}"

    rows = [
        "# 一键训练候选验收（不自动发布）", "",
        "数据、代码和模型指纹见 `evaluation.json`；neutral 阈值各自只用 dev 校准。",
        "冻结 test 仅作最终描述与既定门槛核验，不据此回调参数。", "",
        "|切片|模型|Macro F1|Spearman|非零宏 MAE|辅助 MAE|零维误激活|neutral 误激活|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in (("dev", "旧 dev"), ("old", "冻结旧 test"), ("added", "新增 test"), ("addedDialogue", "新增对白"), ("addedNarration", "新增旁白")):
        for name, display in (("formal", "正式 A"), ("candidate", "候选")):
            metric = report["metrics"][name]["dev"]["all"] if split == "dev" else report["metrics"][name]["test"][split]
            neutral = metric["neutral"]
            rows.append(
                f"|{label}|{display}|{fmt(metric['macroF1'])}|{fmt(metric['macroSpearman'])}|"
                f"{fmt(metric['macroNonzeroMae'])}|{fmt(metric['auxiliary']['mae'])}|"
                f"{fmt(metric['zeroDimensionFalseActivationRate'])}|"
                f"{neutral['falseActivationCount']}/{neutral['count']}|"
            )
    rows.extend(["", "## 冻结质量门", ""])
    for name, passed in report["checks"].items():
        rows.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    rows.extend([
        "", "结论：" + (
            "全部既定门槛通过；仍须用户确认后才可发布。"
            if report["allGatesPassed"] else
            "存在未通过项；保留候选产物，不替换正式 A 或交接包。"
        ), "",
    ])
    return "\n".join(rows)


def evaluate(
    *, formal: Path, candidate: Path, dev: Path, test: Path, output: Path,
    expected_formal_sha256: str, expected_added_test_count: int,
    added_prefix: str = "supp-", device: str = "cuda", batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    if file_sha256(formal / "model.safetensors").lower() != expected_formal_sha256.lower():
        raise ValueError("正式 A 权重哈希变化，拒绝比较")
    sources = {"dev": dev, "test": test}
    examples = {name: load_jsonl_examples(path) for name, path in sources.items()}
    added = [row for row in examples["test"] if row.example_id.startswith(added_prefix)]
    if len(added) != expected_added_test_count or not any(row.sentence_type == "narration" for row in added):
        raise ValueError("新增 test 数量或旁白切片与冻结配置不符")

    results = {}
    test_predictions = {}
    selected_device = torch.device(device)
    for name, checkpoint in (("formal", formal), ("candidate", candidate)):
        model, tokenizer = load_checkpoint(checkpoint)
        model.to(selected_device)
        predictions = {
            split: load_or_predict(
                model, tokenizer, checkpoint, source, examples[split], output / f"{name}-{split}",
                device=selected_device, batch_size=batch_size, max_length=max_length,
            )
            for split, source in sources.items()
        }
        for split in sources:
            write_prediction_rows(output / f"{name}-{split}-predictions.jsonl", examples[split], predictions[split])
        test_predictions[name] = predictions["test"]
        del model
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
        calibration = calibrate_neutral_threshold(
            predictions["dev"]["intensities"], predictions["dev"]["predictedIntensities"],
            minimum_active_recall=0.8,
        )
        threshold = float(calibration["recommended"]["threshold"])
        results[name] = {
            "threshold": threshold,
            "modelSha256": file_sha256(checkpoint / "model.safetensors"),
            "dev": {"all": detailed_metrics(predictions["dev"], threshold)},
            "test": slice_metrics(predictions["test"], examples["test"], threshold, added_prefix),
        }
        write_json(output / f"{name}-threshold-calibration.json", calibration)

    checks = quality_checks(results["formal"]["dev"]["all"], results["candidate"]["dev"]["all"], results["candidate"]["test"])
    write_json(
        output / "added-test-cases.json",
        added_test_cases(examples["test"], test_predictions["formal"], test_predictions["candidate"], added_prefix),
    )
    report = {
        "schema": SCHEMA,
        "inputSha256": {name: file_sha256(path) for name, path in sources.items()},
        "expectedAddedTestCount": expected_added_test_count,
        "addedIdPrefix": added_prefix,
        "selectionPolicy": "dev-only; test descriptive frozen acceptance; no automatic publication",
        "gates": {"devSafety": DEV_SAFETY, "addedTest": ADDED_TEST_GATES},
        "metrics": results,
        "checks": checks,
        "allGatesPassed": all(checks.values()),
    }
    write_json(output / "evaluation.json", report)
    (output / "evaluation.zh-CN.md").write_text(render_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("formal", "candidate", "dev", "test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-formal-sha256", required=True)
    parser.add_argument("--expected-added-test-count", type=int, required=True)
    parser.add_argument("--added-prefix", default="supp-")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    result = evaluate(**vars(args))
    print(json.dumps({"allGatesPassed": result["allGatesPassed"], "checks": result["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
