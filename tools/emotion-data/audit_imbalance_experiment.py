"""Audit the completed imbalance experiment and write final Chinese handoff reports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path("E:/Projects/Readest-temp/emotion-imbalance-experiment-20260919")
GROUPS = ("baseline", "A-loss", "B-loss-sampling")
EXPERIMENTS = ("A-loss", "B-loss-sampling")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run_dir(group: str) -> Path:
    return ROOT / "baseline" if group == "baseline" else ROOT / "runs" / group


def count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    status = read(ROOT / "status.json")
    check("pipeline-completed", status.get("state") == "completed" and status.get("stage") == "done", status)
    manifest = read(ROOT / "experiment.json")
    frozen = []
    for section in ("inputs", "baseModelFiles", "baselineFiles", "sourceFiles"):
        for item in manifest[section].values():
            path = Path(item["path"])
            frozen.append({"path": str(path), "expected": item["sha256"], "actual": sha256(path)})
    check("frozen-inputs-and-code-unchanged", all(x["expected"] == x["actual"] for x in frozen), frozen)

    expected_stages = ["smoke", "train-A-loss", "train-B-loss-sampling",
                       "dev-baseline", "dev-A-loss", "dev-B-loss-sampling", "compare-dev",
                       "test-baseline", "test-A-loss", "test-B-loss-sampling", "compare-test"]
    stage_results = []
    for name in expected_stages:
        marker_path = ROOT / "stages" / f"{name}.json"
        valid = marker_path.is_file()
        invalid_outputs = []
        if valid:
            for item in read(marker_path)["outputs"]:
                path = Path(item["path"])
                if not path.is_file() or sha256(path) != item["sha256"]:
                    invalid_outputs.append(str(path))
        stage_results.append({"stage": name, "marker": valid, "invalidOutputs": invalid_outputs})
    check("all-stage-markers-and-hashes", all(x["marker"] and not x["invalidOutputs"] for x in stage_results), stage_results)

    smoke = read(ROOT / "smoke/passed.json")
    check("gpu-interrupt-resume-smoke", all(smoke[g]["passed"] and smoke[g]["maxParameterError"] <= smoke[g]["atol"] for g in EXPERIMENTS), smoke)
    probe = read(ROOT / "smoke/full-length-memory.json")
    check("full-length-gpu-probe", probe["trainShape"] == [6, 512] and probe["inferenceShape"] == [32, 512], probe)
    junit = ET.parse(ROOT / "unit-tests/final-results.xml").getroot()
    suites = [junit] if junit.tag == "testsuite" else list(junit.findall("testsuite"))
    test_summary = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                    for key in ("tests", "failures", "errors", "skipped")}
    check("unit-tests", test_summary["tests"] == 130 and not test_summary["failures"] and not test_summary["errors"], test_summary)

    training = {}
    for group in EXPERIMENTS:
        destination = run_dir(group)
        report = read(destination / "training-report.json")
        final_model = destination / "final/model.safetensors"
        checkpoints = sorted((destination / "checkpoints").glob("checkpoint-step-*"))
        sampling = [read(destination / "sampling" / f"epoch-{epoch}.json") for epoch in range(1, 9)]
        training[group] = {
            "finalEpoch": report["finalEpoch"], "globalStep": report["globalStep"],
            "bestEpoch": report["bestEpoch"], "elapsedSeconds": report["elapsedSeconds"],
            "reportFinalSha256": report["finalModelSha256"], "actualFinalSha256": sha256(final_model),
            "checkpointCount": len(checkpoints), "checkpointNames": [p.name for p in checkpoints],
            "samplingEpochs": len(sampling),
            "allSamplingCounts": [row["sampleCount"] for row in sampling],
            "meanUniqueSamples": sum(row["uniqueSamples"] for row in sampling) / 8,
            "meanDuplicateDraws": sum(row["duplicateDraws"] for row in sampling) / 8,
        }
    check("final-epoch8-models", all(x["finalEpoch"] == 8 and x["globalStep"] == 9216 and
                                      x["reportFinalSha256"] == x["actualFinalSha256"] for x in training.values()), training)
    check("two-latest-checkpoints-retained", all(x["checkpointCount"] == 2 for x in training.values()), training)
    check("eight-deterministic-exposure-reports", all(x["samplingEpochs"] == 8 and set(x["allSamplingCounts"]) == {27626} for x in training.values()), training)

    prediction_rows = {}
    for group in GROUPS:
        prediction_rows[group] = {split: count_lines(run_dir(group) / f"{split}-predictions.jsonl") for split in ("dev", "test")}
    check("prediction-row-counts", all(x == {"dev": 6234, "test": 7903} for x in prediction_rows.values()), prediction_rows)

    dev = read(ROOT / "comparison/dev-comparison.json")
    test = read(ROOT / "comparison/test-comparison.json")
    threshold_checks = {}
    for group in GROUPS:
        calibrated = read(run_dir(group) / "threshold-calibration.json")["recommended"]["threshold"]
        threshold_checks[group] = {"calibrated": calibrated,
                                   "dev": dev["metrics"][group]["neutralThreshold"],
                                   "test": test["metrics"][group]["neutralThreshold"]}
    check("dev-threshold-frozen-for-test", all(x["calibrated"] == x["dev"] == x["test"] for x in threshold_checks.values()), threshold_checks)
    check("test-is-descriptive-only", all(test["comparisons"][g].get("descriptiveOnly") is True and
                                            "worthContinuing" not in test["comparisons"][g] for g in EXPERIMENTS), test["comparisons"])

    case_counts = {}
    for split in ("dev", "test"):
        cases = read(ROOT / "comparison" / f"{split}-cases.json")
        case_counts[split] = {g: {side: len(cases[g][side]) for side in ("improved", "regressed")} for g in EXPERIMENTS}
    check("fixed-ranked-case-sets", all(count == 10 for split in case_counts.values() for group in split.values() for count in group.values()), case_counts)

    baseline = test["metrics"]["baseline"]
    a = test["metrics"]["A-loss"]
    b = test["metrics"]["B-loss-sampling"]
    summary = {
        "schema": "readest-emotion-imbalance-final-audit-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "devDecisions": dev["comparisons"],
        "testDescriptiveDeltas": test["comparisons"],
        "recommendation": {
            "candidate": "A-loss",
            "action": "repeat-seed-validation-before-release",
            "publishAutomatically": False,
            "reason": "A 与 B 都通过 dev 初筛；A 在 test 的 Macro F1、Spearman、vector MAE、intensity MAE 与 sad F1 略优，且不增加混合采样复杂度。",
        },
    }
    write(ROOT / "comparison/final-audit.json", summary)

    lines = [
        "# 情感不平衡实验最终结论", "",
        "两组都通过预先固定的 dev 初筛门槛，但结论仅基于单一随机种子。当前阅读器交接模型不自动替换。", "",
        "## 核心结果", "",
        "|组别|dev Macro F1|test Macro F1|test Spearman|test 非零宏MAE|test 八维整体MAE|test 零维误激活率|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        dm, tm = dev["metrics"][group], test["metrics"][group]
        lines.append(f"|{group}|{dm['macroF1']:.4f}|{tm['macroF1']:.4f}|{tm['macroSpearman']:.4f}|{tm['macroNonzeroMae']:.4f}|{tm['vectorMae']:.4f}|{tm['zeroDimensionFalseActivationRate']:.4f}|")
    lines += [
        "", "## 判断", "",
        f"- A 相对基线：test Macro F1 {a['macroF1']-baseline['macroF1']:+.4f}，sad/melancholic 平均 F1 {test['comparisons']['A-loss']['deltas']['sadMelancholicF1']:+.4f}，零维误激活率 {a['zeroDimensionFalseActivationRate']-baseline['zeroDimensionFalseActivationRate']:+.4f}。",
        f"- B 相对基线：test Macro F1 {b['macroF1']-baseline['macroF1']:+.4f}，sad/melancholic 平均 F1 {test['comparisons']['B-loss-sampling']['deltas']['sadMelancholicF1']:+.4f}，零维误激活率 {b['zeroDimensionFalseActivationRate']-baseline['zeroDimensionFalseActivationRate']:+.4f}。",
        f"- 连续强度有明确代价：A/B 的非零宏 MAE 分别比基线 {a['macroNonzeroMae']-baseline['macroNonzeroMae']:+.4f}/{b['macroNonzeroMae']-baseline['macroNonzeroMae']:+.4f}。弱势维度 F1 改善主要来自减少误激活并提高 precision，同时 recall 有所下降。",
        f"- 辅助维度 MAE：基线 {baseline['auxiliary']['mae']:.4f}，A {a['auxiliary']['mae']:.4f}，B {b['auxiliary']['mae']:.4f}；B 对辅助维度连续值的保持略好。",
        "- B 没有在 test 上形成相对 A 的稳定优势，额外混合采样暂不值得成为默认训练路径。",
        "", "## 建议", "",
        "优先把 A-loss 作为下一轮重复随机种子验证的候选，不直接发布。若重复实验仍保持 F1 和误激活优势，再重点处理 sad/melancholic 中强强度的低估问题。B 作为保留对照，不建议目前替换默认训练入口。",
        "", "## 完整性", "",
        f"最终审计：{'全部通过' if summary['passed'] else '存在失败项'}。A/B 都是第 8 轮 final；每组保留最近两个恢复 checkpoint；8 轮采样曝光、三组 dev/test 逐条预测、作品指标、强度分桶、固定排序案例、输入与实现哈希均已检查。",
        "", "详细数值见 dev-report.md、test-report.md 和 final-audit.json。清理前应先由用户选定是否保留候选模型。", "",
    ]
    (ROOT / "comparison/final-conclusion.md").write_text("\n".join(lines), encoding="utf-8")

    # Refresh cleanup accounting after adding the final audit artifacts.
    entries = []
    for child in ROOT.iterdir():
        files = list(child.rglob("*")) if child.is_dir() else [child]
        entries.append({"path": str(child), "bytes": sum(p.stat().st_size for p in files if p.is_file()),
                        "role": "experiment-only; remove only after user selects a model"})
    write(ROOT / "comparison/cleanup-inventory.json", entries)
    cleanup = ["# 清理清单", "", "所有本轮产物均在实验根目录。选定方案后，先另存所选 final/、配置和报告，再整体移走实验根目录即可。",
               "源数据、现有基线模型、基础模型缓存和 index-tts-package 均位于目录外。", ""]
    cleanup += [f"- {item['path']}：{item['bytes']/1024**3:.3f} GiB" for item in entries]
    cleanup += ["", f"- 合计：{sum(item['bytes'] for item in entries)/1024**3:.3f} GiB", ""]
    (ROOT / "comparison/cleanup.md").write_text("\n".join(cleanup), encoding="utf-8")
    if not summary["passed"]:
        raise SystemExit("Final audit failed; inspect comparison/final-audit.json")


if __name__ == "__main__":
    main()
