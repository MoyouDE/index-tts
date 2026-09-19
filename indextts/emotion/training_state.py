"""Crash-safe state management for supervised emotion training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from .annotation import file_sha256
from .model import MODEL_WEIGHTS, MacBertEmotionModel, save_checkpoint
from .schema import EmotionExample


CHECKPOINT_SCHEMA = "readest-emotion-checkpoint-v1"
LEGACY_COMPLETION_SCHEMA = "readest-emotion-training-complete-v1"
COMPLETION_SCHEMA = "readest-emotion-training-complete-v2"
CHECKPOINT_COMPLETE = "COMPLETE"
TRAINING_COMPLETE = "training-complete.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_training_fingerprint(
    train_examples: Sequence[EmotionExample],
    dev_examples: Sequence[EmotionExample],
    config: Mapping[str, object],
    input_manifest: Mapping[str, object] | None = None,
) -> str:
    """Fingerprint the exact examples, source hashes, and optimization route."""

    digest = hashlib.sha256()
    digest.update(_canonical_json({"config": dict(config), "inputs": input_manifest or {}}))
    for split, examples in (("train", train_examples), ("dev", dev_examples)):
        digest.update(split.encode("ascii"))
        for example in examples:
            digest.update(_canonical_json(example.as_json()))
    return digest.hexdigest()


def epoch_batch_indices(
    item_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Return a stable per-epoch permutation that can resume at a batch boundary."""

    if item_count <= 0:
        raise ValueError("训练集不能为空")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    generator = torch.Generator().manual_seed(int(seed) + int(epoch) - 1)
    indices = torch.randperm(item_count, generator=generator).tolist()
    return [indices[start : start + batch_size] for start in range(0, item_count, batch_size)]


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class TrainingRunLock:
    """Cross-platform process lock held for the lifetime of one output run."""

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / ".train.lock"
        self._handle = None

    def __enter__(self) -> "TrainingRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                if self.path.stat().st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                    handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(f"输出目录已有训练进程: {self.path.parent}") from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            _canonical_json(
                {
                    "pid": os.getpid(),
                    "startedAt": _utc_now(),
                    "outputDir": self.path.parent.resolve().as_posix(),
                }
            )
        )
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass(frozen=True)
class ResumeState:
    checkpoint_dir: Path
    epoch: int
    batch_index: int
    global_step: int
    history: list[dict[str, object]]
    best_score: float
    epoch_rolling: dict[str, float]
    epoch_batch_count: int
    rng_state: Mapping[str, Any]


def _checkpoint_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints"


def _cleanup_temporary_checkpoints(root: Path) -> None:
    if not root.is_dir():
        return
    for path in root.glob(".checkpoint-step-*.tmp-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _read_checkpoint_manifest(path: Path) -> dict[str, object] | None:
    manifest_path = path / "trainer-state.json"
    if not (path / CHECKPOINT_COMPLETE).is_file() or not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        return None
    hashes = value.get("fileHashes")
    if not isinstance(hashes, dict):
        return None
    for filename in (MODEL_WEIGHTS, "optimizer.pt", "scheduler.pt", "rng.pt"):
        expected = hashes.get(filename)
        candidate = path / filename
        if not isinstance(expected, str) or not candidate.is_file():
            return None
        if file_sha256(candidate) != expected:
            return None
    return value


def find_resume_checkpoint(
    output_dir: str | Path,
    fingerprint: str,
) -> tuple[Path, dict[str, object]] | None:
    root = _checkpoint_root(output_dir)
    if not root.is_dir():
        return None
    _cleanup_temporary_checkpoints(root)
    valid: list[tuple[int, Path, dict[str, object]]] = []
    foreign: list[Path] = []
    for path in root.glob("checkpoint-step-*"):
        if not path.is_dir():
            continue
        manifest = _read_checkpoint_manifest(path)
        if manifest is None:
            continue
        if manifest.get("fingerprint") != fingerprint:
            foreign.append(path)
            continue
        valid.append((int(manifest.get("globalStep", -1)), path, manifest))
    if valid:
        _, path, manifest = max(valid, key=lambda item: item[0])
        return path, manifest
    if foreign:
        raise ValueError("输出目录中的 checkpoint 与当前数据或训练配置不匹配")
    return None


def save_training_checkpoint(
    output_dir: str | Path,
    *,
    model: MacBertEmotionModel,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    rng_state: Mapping[str, Any],
    fingerprint: str,
    epoch: int,
    batch_index: int,
    global_step: int,
    history: Sequence[Mapping[str, object]],
    best_score: float,
    epoch_rolling: Mapping[str, float],
    epoch_batch_count: int,
    keep_checkpoints: int,
) -> Path:
    root = _checkpoint_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _cleanup_temporary_checkpoints(root)
    name = f"checkpoint-step-{global_step:010d}"
    final = root / name
    existing = _read_checkpoint_manifest(final) if final.is_dir() else None
    if existing is not None and existing.get("fingerprint") == fingerprint:
        return final
    if final.exists():
        raise RuntimeError(f"checkpoint 目标已存在但不完整: {final}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=root))
    try:
        save_checkpoint(model, tokenizer, temporary)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(dict(rng_state), temporary / "rng.pt")
        hashes = {
            filename: file_sha256(temporary / filename)
            for filename in (MODEL_WEIGHTS, "optimizer.pt", "scheduler.pt", "rng.pt")
        }
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "fingerprint": fingerprint,
            "epoch": int(epoch),
            "batchIndex": int(batch_index),
            "globalStep": int(global_step),
            "history": [dict(item) for item in history],
            "bestScore": None if best_score == float("-inf") else float(best_score),
            "epochRolling": {key: float(value) for key, value in epoch_rolling.items()},
            "epochBatchCount": int(epoch_batch_count),
            "createdAt": _utc_now(),
            "fileHashes": hashes,
        }
        (temporary / "trainer-state.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / CHECKPOINT_COMPLETE).write_text("ok\n", encoding="ascii")
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    _prune_checkpoints(root, fingerprint, keep_checkpoints)
    return final


def _prune_checkpoints(root: Path, fingerprint: str, keep_checkpoints: int) -> None:
    if keep_checkpoints <= 0:
        return
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-step-*"):
        manifest = _read_checkpoint_manifest(path)
        if manifest is not None and manifest.get("fingerprint") == fingerprint:
            candidates.append((int(manifest["globalStep"]), path))
    for _, path in sorted(candidates, reverse=True)[keep_checkpoints:]:
        shutil.rmtree(path)


def restore_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_fingerprint: str,
    model: MacBertEmotionModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> ResumeState:
    source = Path(checkpoint_dir)
    manifest = _read_checkpoint_manifest(source)
    if manifest is None:
        raise ValueError(f"checkpoint 损坏或不完整: {source}")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint 与当前数据或训练配置不匹配")
    missing, unexpected = model.load_state_dict(load_file(source / MODEL_WEIGHTS), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint 模型 ABI 不匹配: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(
        torch.load(source / "optimizer.pt", map_location=device, weights_only=False)
    )
    scheduler.load_state_dict(
        torch.load(source / "scheduler.pt", map_location="cpu", weights_only=False)
    )
    rng_state = torch.load(source / "rng.pt", map_location="cpu", weights_only=False)
    return ResumeState(
        checkpoint_dir=source,
        epoch=int(manifest["epoch"]),
        batch_index=int(manifest["batchIndex"]),
        global_step=int(manifest["globalStep"]),
        history=[dict(item) for item in manifest.get("history", [])],
        best_score=(
            float("-inf")
            if manifest.get("bestScore") is None
            else float(manifest["bestScore"])
        ),
        epoch_rolling={
            str(key): float(value)
            for key, value in dict(manifest.get("epochRolling", {})).items()
        },
        epoch_batch_count=int(manifest.get("epochBatchCount", 0)),
        rng_state=rng_state,
    )


def read_completed_report(
    output_dir: str | Path,
    fingerprint: str,
) -> dict[str, object] | None:
    target = Path(output_dir)
    marker = target / TRAINING_COMPLETE
    if not marker.is_file():
        return None
    try:
        completion = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"训练完成标记损坏: {marker}") from exc
    schema = completion.get("schema")
    if schema not in {LEGACY_COMPLETION_SCHEMA, COMPLETION_SCHEMA}:
        raise ValueError(f"训练完成标记 schema 无效: {marker}")
    if completion.get("fingerprint") != fingerprint:
        raise ValueError("既有训练结果与当前数据或训练配置不匹配")
    report_path = target / "training-report.json"
    if not report_path.is_file():
        raise ValueError("训练完成标记存在，但训练报告缺失")
    if schema == LEGACY_COMPLETION_SCHEMA:
        best_weights = target / "best" / MODEL_WEIGHTS
        if not best_weights.is_file():
            raise ValueError("训练完成标记存在，但最佳模型缺失")
        if completion.get("bestModelSha256") != file_sha256(best_weights):
            raise ValueError("训练完成标记对应的最佳模型哈希不匹配")
    else:
        final_weights = target / "final" / MODEL_WEIGHTS
        if not final_weights.is_file():
            raise ValueError("训练完成标记存在，但最终轮模型缺失")
        if completion.get("finalModelSha256") != file_sha256(final_weights):
            raise ValueError("训练完成标记对应的最终轮模型哈希不匹配")
        best_hash = completion.get("bestModelSha256")
        best_weights = target / "best" / MODEL_WEIGHTS
        if best_hash is not None and (
            not best_weights.is_file() or best_hash != file_sha256(best_weights)
        ):
            raise ValueError("训练完成标记对应的最佳模型哈希不匹配")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("training-report.json 格式无效")
    report["alreadyCompleted"] = True
    return report


def mark_training_complete(
    output_dir: str | Path,
    *,
    fingerprint: str,
    global_step: int,
    best_score: float,
) -> Path:
    target = Path(output_dir)
    marker = target / TRAINING_COMPLETE
    best_weights = target / "best" / MODEL_WEIGHTS
    value = {
        "schema": COMPLETION_SCHEMA,
        "fingerprint": fingerprint,
        "globalStep": int(global_step),
        "bestSelectionScore": float(best_score),
        "bestModelSha256": file_sha256(best_weights) if best_weights.is_file() else None,
        "finalModelSha256": file_sha256(target / "final" / MODEL_WEIGHTS),
        "completedAt": _utc_now(),
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    return marker
