"""One-click, versioned emotion candidate training; never publishes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from indextts.emotion.training_state import TrainingRunLock

DEFAULT_PROFILE = Path(__file__).with_name("one-click-profile.json")
SCHEMA = "readest-emotion-one-click-run-v1"
CODE_PATHS = (
    "indextts/emotion/cli.py",
    "indextts/emotion/dataset.py",
    "indextts/emotion/imbalance.py",
    "indextts/emotion/model.py",
    "indextts/emotion/schema.py",
    "indextts/emotion/train.py",
    "indextts/emotion/training_preflight.py",
    "indextts/emotion/training_state.py",
    "tools/emotion-data/evaluate_one_click.py",
    "tools/emotion-data/run_one_click.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def path_from_profile(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def row_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def build_manifest(profile_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    profile = read_json(profile_path)
    if profile.get("schema") != "readest-emotion-one-click-profile-v1":
        raise ValueError("训练 profile schema 不匹配")
    method = profile["method"]
    if method["trainingObjective"] != "balanced-regression-v1" or method["samplingMode"] != "uniform":
        raise ValueError("正式一键 BAT 只允许已验证的 balanced-regression-v1 + uniform")
    data = path_from_profile(profile["datasetDir"])
    output = path_from_profile(profile["outputDir"])
    formal = path_from_profile(profile["formalCheckpoint"])
    base_model = path_from_profile(profile["baseModel"])
    if output == formal or output in formal.parents or formal in output.parents:
        raise ValueError("候选输出目录不得与正式模型目录重叠")
    if output == data or output in data.parents or data in output.parents:
        raise ValueError("候选输出目录不得与数据目录重叠")
    inputs = {name: data / f"{name}.jsonl" for name in ("train", "dev", "test")}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = int(profile[f"expected{name.capitalize()}Rows"])
        actual = row_count(path)
        if actual != expected:
            raise ValueError(f"{name} 行数不符: expected={expected}, actual={actual}")
        if sha256(path).lower() != profile["inputSha256"][name].lower():
            raise ValueError(f"{name} 文件与 profile 冻结哈希不符")
    weights = formal / "model.safetensors"
    if sha256(weights).lower() != profile["formalModelSha256"].lower():
        raise ValueError("正式 A 权重与 profile 冻结哈希不符")
    base_weights = base_model / "pytorch_model.bin"
    if sha256(base_weights).lower() != profile["baseWeightsSha256"].lower():
        raise ValueError("基础 MacBERT 权重与 profile 冻结哈希不符")
    base_files = (
        "pytorch_model.bin", "config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json", "added_tokens.json",
    )
    manifest = {
        "schema": SCHEMA,
        "profilePath": str(profile_path.resolve()),
        "profileSha256": sha256(profile_path),
        "inputs": {name: {"path": str(path), "rows": row_count(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "formalModel": {"path": str(weights), "sha256": sha256(weights)},
        "baseModel": {"path": str(base_model), "files": {
            name: sha256(base_model / name) for name in base_files
        }},
        "codeSha256": {relative: sha256(REPO / relative) for relative in CODE_PATHS},
        "method": method,
        "resourceMinimums": profile["resources"],
    }
    return profile, manifest, data, output, formal


def ensure_manifest(output: Path, manifest: dict) -> None:
    existing = output / "run-manifest.json"
    if existing.is_file():
        if read_json(existing) != manifest:
            raise ValueError("候选目录中的冻结输入/代码/训练配置与本次不一致；拒绝续训")
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"候选目录非空但无 run-manifest.json: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(existing, manifest)


def status(output: Path, stage: str, *, error: str | None = None) -> None:
    write_json(output / "status.json", {
        "schema": SCHEMA,
        "stage": stage,
        "error": error,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "published": False,
    })


def run_command(args: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(args, cwd=REPO, env=env, check=True)


def show_stage(index: int, title: str) -> None:
    print(f"\n[{index}/3] {title}...\n", flush=True)


def verify_training(output: Path, expected_epochs: int) -> None:
    run = output / "run"
    complete = read_json(run / "training-complete.json")
    report = read_json(run / "training-report.json")
    final = run / "final" / "model.safetensors"
    actual = sha256(final)
    if (
        int(report["finalEpoch"]) != expected_epochs
        or complete["finalModelSha256"] != actual
        or report["finalModelSha256"] != actual
    ):
        raise ValueError("最终轮模型、报告或完成标记不一致")


def run(profile_path: Path, *, validate_only: bool = False) -> dict:
    profile, manifest, data, output, formal = build_manifest(profile_path)
    base_model = path_from_profile(profile["baseModel"])
    if validate_only:
        print(json.dumps({"output": str(output), "manifest": manifest}, ensure_ascii=False, indent=2))
        return manifest
    ensure_manifest(output, manifest)
    pipeline_lock = TrainingRunLock(output)
    pipeline_lock.__enter__()
    method = profile["method"]
    resources = profile["resources"]
    print(f"  Data:          {data}", flush=True)
    print(f"  Output:        {output}", flush=True)
    print(f"  Epochs:        {method['epochs']}", flush=True)
    print(f"  Batch Size:    {method['batchSize']}", flush=True)
    print(f"  Accumulation:  {method['gradientAccumulation']}", flush=True)
    print(f"  Max Length:    {method['maxLength']}", flush=True)
    python = sys.executable
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    base = [python, "-X", "utf8", "-m", "indextts.emotion.cli"]
    try:
        show_stage(1, "预检")
        status(output, "preflight")
        run_command(base + [
            "preflight-training", "--train", str(data / "train.jsonl"),
            "--dev", str(data / "dev.jsonl"), "--test", str(data / "test.jsonl"),
            "--output", str(output / "preflight.json"),
            "--max-length", str(method["maxLength"]), "--require-complete-context",
            "--base-model", str(base_model), "--verify-base-weights", "--require-cuda",
            "--minimum-free-vram-gib", str(resources["minimumFreeVramGiB"]),
            "--minimum-free-disk-gib", str(resources["minimumFreeDiskGiB"]),
            "--progress",
        ], env=env)
        if read_json(output / "preflight.json").get("preflightPassed") is not True:
            raise ValueError("数据预检未通过")

        show_stage(2, "训练")
        status(output, "training")
        run_command(base + [
            "train", "--train", str(data / "train.jsonl"), "--dev", str(data / "dev.jsonl"),
            "--output", str(output / "run"), "--training-objective", method["trainingObjective"],
            "--base-model", str(base_model),
            "--sampling-mode", method["samplingMode"], "--epochs", str(method["epochs"]),
            "--batch-size", str(method["batchSize"]),
            "--gradient-accumulation", str(method["gradientAccumulation"]),
            "--learning-rate", str(method["learningRate"]),
            "--head-learning-rate", str(method["headLearningRate"]),
            "--intensity-loss-weight", str(method["intensityLossWeight"]),
            "--neutral-loss-weight", str(method["neutralLossWeight"]),
            "--max-length", str(method["maxLength"]), "--seed", str(method["seed"]),
            "--device", "cuda", "--resume", "auto",
            "--checkpoint-steps", str(method["checkpointSteps"]),
            "--keep-checkpoints", str(method["keepCheckpoints"]), "--progress",
        ], env=env)
        verify_training(output, int(method["epochs"]))

        show_stage(3, "评估")
        status(output, "evaluation")
        run_command([
            python, "-X", "utf8", str(REPO / "tools/emotion-data/evaluate_one_click.py"),
            "--formal", str(formal), "--candidate", str(output / "run" / "final"),
            "--dev", str(data / "dev.jsonl"), "--test", str(data / "test.jsonl"),
            "--output", str(output / "evaluation"),
            "--expected-formal-sha256", profile["formalModelSha256"],
            "--expected-added-test-count", str(profile["expectedAddedTestRows"]),
            "--added-prefix", profile["addedIdPrefix"], "--device", "cuda",
            "--batch-size", "32", "--max-length", str(method["maxLength"]),
        ], env=env)
        evaluation = read_json(output / "evaluation" / "evaluation.json")
        status(output, "completed")
        print("Candidate:", output / "run" / "final", flush=True)
        print("Report:", output / "evaluation" / "evaluation.zh-CN.md", flush=True)
        print("All frozen gates passed:", evaluation["allGatesPassed"], flush=True)
        print("Formal A and reader handoff remain untouched.", flush=True)
        return evaluation
    except KeyboardInterrupt:
        status(output, "interrupted", error="KeyboardInterrupt; rerun the BAT to resume")
        raise
    except Exception as exc:
        status(output, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pipeline_lock.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(args.profile.resolve(), validate_only=args.validate_only)


if __name__ == "__main__":
    main()
