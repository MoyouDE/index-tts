"""Promote the selected imbalance experiment into the stable production layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from indextts.emotion.annotation import file_sha256
from indextts.emotion.imbalance import ImbalanceConfig
from indextts.emotion.schema import load_jsonl_examples
from indextts.emotion.training_state import build_training_fingerprint


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def promote(experiment_root: Path, output: Path) -> dict:
    experiment_root = experiment_root.resolve()
    output = output.resolve()
    audit_path = experiment_root / "comparison" / "final-audit.json"
    audit = read_json(audit_path)
    recommendation = audit.get("recommendation")
    if audit.get("passed") is not True or not isinstance(recommendation, dict):
        raise ValueError("Experiment final audit did not pass")
    candidate = str(recommendation.get("candidate", "")).strip()
    if candidate != "A-loss":
        raise ValueError(f"Expected A-loss as selected candidate, got {candidate!r}")

    source = experiment_root / "runs" / candidate
    source_report_path = source / "training-report.json"
    source_report = read_json(source_report_path)
    final_model = source / "final" / "model.safetensors"
    if source_report.get("finalEpoch") != 8:
        raise ValueError("Selected model is not the fixed epoch-8 checkpoint")
    if sha256(final_model) != source_report.get("finalModelSha256"):
        raise ValueError("Selected final checkpoint hash does not match its training report")

    dev = read_json(source / "dev-metrics.json")
    test = read_json(source / "test-metrics.json")
    if int(dev.get("sampleCount", 0)) != 6234 or int(test.get("sampleCount", 0)) != 7903:
        raise ValueError("Selected metrics do not cover the frozen dev/test splits")

    if output.exists():
        raise FileExistsError(f"Promotion target already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copytree(source / "final", stage / "final")
        for name in (
            "weights.json",
            "dev-metrics.json",
            "test-metrics.json",
            "threshold-calibration.json",
        ):
            shutil.copy2(source / name, stage / name)
        shutil.copy2(source_report_path, stage / "source-training-report.json")
        shutil.copy2(audit_path, stage / "experiment-audit.json")
        shutil.copy2(experiment_root / "comparison" / "final-conclusion.md", stage / "experiment-conclusion.md")

        config = read_json(source / "config.json")
        objective_value = config.pop("trainingObjective", None)
        if objective_value is None:
            objective_value = config.pop("experiment")
        objective = dict(objective_value)
        objective["implementation"] = "balanced-regression-v1"
        config["trainingObjective"] = objective
        write_json(stage / "config.json", config)

        data = REPO / "data" / "emotion" / "dialogue-stage-20260917-deepseek-v3-final-41763"
        train_path, dev_path = data / "train.jsonl", data / "dev.jsonl"
        input_manifest = {
            split: [{
                "path": path.resolve().as_posix(),
                "sha256": file_sha256(path),
                "sizeBytes": path.stat().st_size,
            }]
            for split, path in (("train", train_path), ("dev", dev_path))
        }
        fingerprint = build_training_fingerprint(
            load_jsonl_examples(train_path),
            load_jsonl_examples(dev_path),
            config,
            input_manifest,
        )

        report = dict(source_report)
        objective_value = report.pop("trainingObjective", None)
        if objective_value is None:
            objective_value = report.pop("experiment")
        objective = dict(objective_value)
        objective["implementation"] = "balanced-regression-v1"
        report["schemaVersion"] = 2
        report["trainingObjective"] = objective
        report["inputFingerprint"] = fingerprint
        report["promotionSource"] = {
            "candidate": candidate,
            "experimentRoot": str(experiment_root),
            "sourceTrainingReportSha256": sha256(source_report_path),
            "experimentAuditSha256": sha256(audit_path),
        }
        write_json(stage / "training-report.json", report)
        write_json(stage / "training-complete.json", {
            "schema": "readest-emotion-training-complete-v2",
            "fingerprint": fingerprint,
            "globalStep": int(report["globalStep"]),
            "bestSelectionScore": float(report["bestSelectionScore"]),
            "bestModelSha256": None,
            "finalModelSha256": report["finalModelSha256"],
            "completedAt": read_json(source / "training-complete.json")["completedAt"],
            "promotedFrom": str(source),
        })

        tracked = [path for path in stage.rglob("*") if path.is_file()]
        manifest = {
            "schema": "readest-emotion-model-promotion-v1",
            "selectedCandidate": candidate,
            "fixedEpoch": 8,
            "neutralThreshold": float(test["neutralThreshold"]),
            "metrics": {
                "macroF1": float(test["macroF1"]),
                "macroSpearman": float(test["macroSpearman"]),
                "intensityMae": float(test["intensityMae"]),
                "vectorMae": float(test["vectorMae"]),
                "sampleCount": int(test["sampleCount"]),
            },
            "files": {
                path.relative_to(stage).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(tracked)
            },
        }
        write_json(stage / "promotion-manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.experiment_root, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
