"""Pure checks for the versioned one-click candidate route."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / "emotion-data" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_one_click")
evaluator = load_tool("evaluate_one_click")


def make_profile(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    base = tmp_path / "base"
    formal = tmp_path / "formal"
    for directory in (data, base, formal):
        directory.mkdir()
    for name in ("train", "dev", "test"):
        (data / f"{name}.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    for name in (
        "pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json", "added_tokens.json",
    ):
        (base / name).write_bytes(name.encode("ascii"))
    (formal / "model.safetensors").write_bytes(b"formal")
    profile = {
        "schema": "readest-emotion-one-click-profile-v1",
        "datasetDir": "data", "outputDir": "candidate", "formalCheckpoint": "formal",
        "formalModelSha256": runner.sha256(formal / "model.safetensors"),
        "baseModel": "base", "baseWeightsSha256": runner.sha256(base / "pytorch_model.bin"),
        "expectedTrainRows": 1, "expectedDevRows": 1, "expectedTestRows": 1,
        "inputSha256": {
            name: runner.sha256(data / f"{name}.jsonl") for name in ("train", "dev", "test")
        },
        "method": {
            "trainingObjective": "balanced-regression-v1", "samplingMode": "uniform",
            "maxLength": 512, "epochs": 8, "batchSize": 6,
            "gradientAccumulation": 4, "learningRate": "2e-5",
            "headLearningRate": "1e-4", "intensityLossWeight": 0.7,
            "neutralLossWeight": 3.0, "seed": 20260829,
            "checkpointSteps": 200, "keepCheckpoints": 2,
        },
        "resources": {"minimumFreeVramGiB": 8.5, "minimumFreeDiskGiB": 10},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def test_manifest_freezes_inputs_code_and_method(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "CODE_PATHS", ())
    profile = make_profile(tmp_path)
    _, manifest, _, output, _ = runner.build_manifest(profile)
    runner.ensure_manifest(output, manifest)
    runner.ensure_manifest(output, manifest)
    assert manifest["inputs"]["train"]["rows"] == 1
    (tmp_path / "data" / "train.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="冻结哈希"):
        runner.build_manifest(profile)
    (tmp_path / "data" / "train.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    altered = json.loads(profile.read_text(encoding="utf-8"))
    altered["method"]["epochs"] = 3
    profile.write_text(json.dumps(altered), encoding="utf-8")
    _, changed, *_ = runner.build_manifest(profile)
    with pytest.raises(ValueError, match="冻结输入"):
        runner.ensure_manifest(output, changed)


def test_manifest_rejects_formal_output_overlap_and_unvalidated_method(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "CODE_PATHS", ())
    path = make_profile(tmp_path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    profile["outputDir"] = "formal/new-run"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="正式模型目录重叠"):
        runner.build_manifest(path)
    profile["outputDir"] = "candidate"
    profile["method"]["trainingObjective"] = "positive-aware-regression-v2"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="只允许已验证"):
        runner.build_manifest(path)


def metric(f1=.6, spearman=.55, zero=.02, aux=.15):
    return {
        "macroF1": f1, "macroSpearman": spearman,
        "zeroDimensionFalseActivationRate": zero,
        "auxiliary": {"mae": aux},
    }


def test_frozen_quality_checks_are_not_weakened():
    test = {
        "added": {
            "macroF1": .48, "macroSpearman": .45,
            "neutralFalseActivationRate": .5, "intensityMae": .2,
        },
        "addedNarration": {"macroF1": .45},
    }
    checks = evaluator.quality_checks(metric(), metric(f1=.59, spearman=.54, zero=.03, aux=.16), test)
    assert all(checks.values())
    failed = evaluator.quality_checks(metric(), metric(f1=.589), test)
    assert not failed["oldDevMacroF1"]
    test["addedNarration"]["macroF1"] = .449
    failed = evaluator.quality_checks(metric(), metric(), test)
    assert not failed["addedNarrationMacroF1"]


def test_verify_training_requires_exact_final_epoch_and_hash(tmp_path):
    run = tmp_path / "run"
    (run / "final").mkdir(parents=True)
    weights = run / "final" / "model.safetensors"
    weights.write_bytes(b"candidate")
    digest = runner.sha256(weights)
    (run / "training-complete.json").write_text(json.dumps({"finalModelSha256": digest}), encoding="utf-8")
    (run / "training-report.json").write_text(json.dumps({"finalEpoch": 8, "finalModelSha256": digest}), encoding="utf-8")
    runner.verify_training(tmp_path, 8)
    with pytest.raises(ValueError, match="最终轮"):
        runner.verify_training(tmp_path, 7)


def test_one_click_stops_when_preflight_report_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "CODE_PATHS", ())
    profile = make_profile(tmp_path)
    commands = []

    def fake_command(args, *, env):
        commands.append(args)
        assert "preflight-training" in args
        assert "--progress" in args
        runner.write_json(tmp_path / "candidate" / "preflight.json", {"preflightPassed": False})

    monkeypatch.setattr(runner, "run_command", fake_command)
    with pytest.raises(ValueError, match="数据预检未通过"):
        runner.run(profile)
    assert len(commands) == 1
    assert runner.read_json(tmp_path / "candidate" / "status.json")["stage"] == "failed"


def test_one_click_stage_headings_match_training_bat_style(capsys):
    for index, title in enumerate(("预检", "训练", "评估"), 1):
        runner.show_stage(index, title)
    output = capsys.readouterr().out
    assert "[1/3] 预检" in output
    assert "[2/3] 训练" in output
    assert "[3/3] 评估" in output


def test_evaluation_rows_and_added_cases_keep_id_and_work(tmp_path):
    rows = [
        SimpleNamespace(example_id="supp-a", work_id="work-a", sentence_type="dialogue"),
        SimpleNamespace(example_id="supp-b", work_id="work-b", sentence_type="narration"),
    ]
    labels = np.zeros((2, 8), dtype=np.float32)
    labels[:, 0] = 0.5
    formal_vectors = labels.copy()
    formal_vectors[0, 0] = 0.2
    formal_vectors[1, 0] = 0.4
    candidate_vectors = labels.copy()
    candidate_vectors[0, 0] = 0.4
    candidate_vectors[1, 0] = 0.1

    def predictions(vectors):
        return {
            "labels": labels, "labelMasks": np.ones_like(labels), "vectors": vectors,
            "intensities": np.array([[0.5], [0.5]], dtype=np.float32),
            "predictedIntensities": np.array([[0.4], [0.3]], dtype=np.float32),
        }

    formal = predictions(formal_vectors)
    candidate = predictions(candidate_vectors)
    output = tmp_path / "predictions.jsonl"
    evaluator.write_prediction_rows(output, rows, candidate)
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [(row["id"], row["workId"]) for row in written] == [
        ("supp-a", "work-a"), ("supp-b", "work-b"),
    ]
    assert len(written[0]["predictions"]) == 8
    cases = evaluator.added_test_cases(rows, formal, candidate, "supp-", limit=1)
    assert cases["improved"][0]["id"] == "supp-a"
    assert cases["degraded"][0]["id"] == "supp-b"
