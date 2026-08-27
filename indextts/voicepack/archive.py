"""Deterministic, pickle-free ``.ivp`` archive support."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from .schema import (
    CONDITIONING_ABI,
    DISCLAIMER_NAME,
    LICENSE_NAME,
    LICENSE_ZH_NAME,
    MANIFEST_NAME,
    MAX_PACK_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    REQUIRED_FILES,
    TENSORS_NAME,
    TENSOR_RULES,
    VoicePackSchemaError,
    validate_manifest,
)


class VoicePackError(ValueError):
    """Raised when a voice pack cannot be safely loaded or verified."""


@dataclass(frozen=True)
class VoicePack:
    path: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]

    @property
    def voice_id(self) -> str:
        return self.manifest["voiceId"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VoicePackError(f"JSON 含重复字段: {key}")
        result[key] = value
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def tensor_manifest(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    if set(tensors) != set(TENSOR_RULES):
        raise VoicePackError("张量集合与 conditioning ABI 不匹配")
    specs: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.layout != torch.strided or tensor.is_sparse:
            raise VoicePackError(f"张量 {name} 必须为稠密 strided tensor")
        specs[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    return specs


def write_voicepack(
    output_path: os.PathLike[str] | str,
    manifest: dict[str, Any],
    tensors: Mapping[str, torch.Tensor],
    license_files: Mapping[str, bytes],
) -> Path:
    expected_licenses = {LICENSE_NAME, LICENSE_ZH_NAME, DISCLAIMER_NAME}
    if set(license_files) != expected_licenses:
        raise VoicePackError("许可文件集合不完整")

    normalized = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(tensors.items())
    }
    tensor_bytes = save_safetensors(normalized, metadata={"conditioningAbi": CONDITIONING_ABI})
    payloads = {TENSORS_NAME: tensor_bytes, **dict(license_files)}
    manifest = dict(manifest)
    manifest["tensors"] = tensor_manifest(normalized)
    manifest["files"] = {name: _sha256(data) for name, data in sorted(payloads.items())}
    try:
        validate_manifest(manifest)
    except VoicePackSchemaError as exc:
        raise VoicePackError(str(exc)) from exc
    payloads[MANIFEST_NAME] = _json_bytes(manifest)

    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".ivp":
        destination = destination.with_suffix(".ivp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=False, compresslevel=9) as archive:
            for name in sorted(payloads):
                archive.writestr(_zip_info(name), payloads[name])
        if temporary.stat().st_size > MAX_PACK_BYTES:
            raise VoicePackError(
                f"音色包超过 {MAX_PACK_BYTES // (1024 * 1024)} MiB 限制: {temporary.stat().st_size} bytes"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_voicepack(
    pack_path: os.PathLike[str] | str,
    *,
    expected_model_fingerprint: str | None = None,
) -> VoicePack:
    path = Path(pack_path).resolve()
    if not path.is_file():
        raise VoicePackError(f"音色包不存在: {path}")
    if path.stat().st_size > MAX_PACK_BYTES:
        raise VoicePackError("音色包超过大小限制")

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise VoicePackError("ZIP 含重复路径")
            for name in names:
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts or "\\" in name or ":" in name:
                    raise VoicePackError(f"ZIP 含不安全路径: {name}")
            if set(names) != REQUIRED_FILES:
                raise VoicePackError("ZIP 文件集合与 schema v1 不匹配")
            if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
                raise VoicePackError("ZIP 解压后大小超过限制")
            raw = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise VoicePackError(f"无法读取音色包: {exc}") from exc

    try:
        manifest = json.loads(raw[MANIFEST_NAME].decode("utf-8"), object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoicePackError(f"manifest.json 无效: {exc}") from exc
    if not isinstance(manifest, dict):
        raise VoicePackError("manifest.json 顶层必须是对象")
    try:
        validate_manifest(manifest)
    except VoicePackSchemaError as exc:
        raise VoicePackError(str(exc)) from exc

    for name, expected in manifest["files"].items():
        actual = _sha256(raw[name])
        if actual != expected:
            raise VoicePackError(f"文件哈希不匹配: {name}")
    if expected_model_fingerprint and manifest["sourceModelFingerprint"] != expected_model_fingerprint:
        raise VoicePackError("音色包源模型指纹与运行时不匹配")

    try:
        tensors = load_safetensors(raw[TENSORS_NAME])
    except Exception as exc:
        raise VoicePackError(f"conditioning.safetensors 无效: {exc}") from exc
    actual_specs = tensor_manifest(tensors)
    if actual_specs != manifest["tensors"]:
        raise VoicePackError("实际张量与 manifest 描述不匹配")
    try:
        validate_manifest(manifest)
    except VoicePackSchemaError as exc:
        raise VoicePackError(str(exc)) from exc
    return VoicePack(path=path, manifest=manifest, tensors=tensors)
