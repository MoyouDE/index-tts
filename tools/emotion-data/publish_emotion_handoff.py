"""Validate and atomically replace the reader emotion-model handoff directory."""
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

from indextts.emotion.release import validate_release_approval


README = """# Readest 情感模型交接包

这是阅读器端可直接接入的正式 FP32 ONNX 情感模型。

- 版本：`20260919-deepseek-v3-41763-balanced-v1`
- 正式训练方案：温和维度加权组成 BCE + 八维 Smooth L1，均匀抽样
- 固定模型：第 8 轮 `final`
- neutral 阈值：`0.355`（仅用 dev 校准）
- 测试集：7,903 条
- Macro F1：`0.6489`
- Macro Spearman：`0.5884`
- 八维向量 MAE：`0.0557`
- 总强度 MAE：`0.0917`

运行时 ABI 为 `readest-emotion-v3`，最大长度 512。`emotion_model.json` 固定标签顺序、上下文策略、special token ID、质量证据摘要和所有包内文件哈希。

接入时复制 manifest 列出的模型与 tokenizer 文件即可；`install-emotion-local.ps1` 会在安装前复核状态、ABI 证据和 SHA-256。
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(source: Path) -> dict:
    manifest_path = source / "emotion_model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("releaseStatus") != "approved":
        raise ValueError("Only an approved model may replace the handoff package")
    validate_release_approval(manifest.get("releaseApproval"))
    files = manifest.get("files")
    if not isinstance(files, dict) or "emotion.onnx" not in files or "tokenizer.json" not in files:
        raise ValueError("Manifest file inventory is incomplete")
    for name, metadata in files.items():
        if Path(name).name != name:
            raise ValueError(f"Unsafe manifest filename: {name}")
        path = source / name
        if not path.is_file() or path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"Missing or truncated source file: {name}")
        if sha256(path) != str(metadata["sha256"]).lower():
            raise ValueError(f"Source hash mismatch: {name}")
    return manifest


def publish(source: Path, target: Path, backup: Path) -> dict:
    source, target, backup = source.resolve(), target.resolve(), backup.resolve()
    manifest = load_manifest(source)
    if backup.exists():
        raise FileExistsError(f"Backup target already exists: {backup}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    moved_old = False
    try:
        for name in [*manifest["files"], "emotion_model.json"]:
            shutil.copy2(source / name, stage / name)
        (stage / "README.md").write_text(README, encoding="utf-8")
        load_manifest(stage)
        if target.exists():
            os.replace(target, backup)
            moved_old = True
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if moved_old and not target.exists():
            os.replace(backup, target)
        raise
    return {
        "version": manifest["version"],
        "target": str(target),
        "backup": str(backup) if moved_old else None,
        "modelSha256": manifest["files"]["emotion.onnx"]["sha256"],
        "sourceCheckpointSha256": manifest["sourceCheckpointSha256"],
        "fileCount": len([path for path in target.iterdir() if path.is_file()]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.target, args.backup), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
