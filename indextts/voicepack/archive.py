"""Deterministic, pickle-free ``.ivp`` archive support."""

from __future__ import annotations

import hashlib
import copy
import io
import json
import os
import zipfile
import uuid
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
    DUAL_FILES, DUAL_VARIANTS, BF16_MANIFEST_NAME, BF16_TENSORS_NAME,
)


class VoicePackError(ValueError):
    """Raised when a voice pack cannot be safely loaded or verified."""


@dataclass(frozen=True)
class VoicePack:
    path: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    available_profiles: tuple[str, ...] = ()
    container_manifest: dict[str, Any] | None = None

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
        if not torch.isfinite(tensor).all().item():
            raise VoicePackError(f"Non-finite voice tensor: {name}")
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
    profile: str | None = None,
) -> VoicePack:
    path = Path(pack_path).resolve()
    if not path.is_file():
        raise VoicePackError(f"音色包不存在: {path}")
    if path.stat().st_size > MAX_PACK_BYTES * 2:
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
            if set(names) not in (REQUIRED_FILES, DUAL_FILES):
                raise VoicePackError("ZIP 文件集合与音色包 schema 不匹配")
            if path.stat().st_size > MAX_PACK_BYTES * (2 if set(names) == DUAL_FILES else 1):
                raise VoicePackError("音色包超过大小限制")
            if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES * (2 if set(names) == DUAL_FILES else 1):
                raise VoicePackError("ZIP 解压后大小超过限制")
            raw = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise VoicePackError(f"无法读取音色包: {exc}") from exc

    return _load_payloads(path, raw, expected_model_fingerprint, profile)


def _load_payloads(path, raw, expected_model_fingerprint=None, profile=None):
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

    expected_files = DUAL_FILES if manifest['schemaVersion'] == 4 else REQUIRED_FILES
    if set(raw) != expected_files:
        raise VoicePackError("ZIP 文件集合与声明的 schema 不匹配")

    for name, expected in manifest["files"].items():
        actual = _sha256(raw[name])
        if actual != expected:
            raise VoicePackError(f"文件哈希不匹配: {name}")
    if expected_model_fingerprint and manifest["sourceModelFingerprint"] != expected_model_fingerprint:
        raise VoicePackError("音色包源模型指纹与运行时不匹配")

    if manifest['schemaVersion'] == 4:
        if profile is not None and profile not in DUAL_VARIANTS:
            raise VoicePackError("未知音色精度")
        common = {n:raw[n] for n in (LICENSE_NAME, LICENSE_ZH_NAME, DISCLAIMER_NAME)}
        fp32 = copy.deepcopy(manifest)
        fp32.pop('variants')
        fp32['schemaVersion'] = 3 if 'referenceSelection' in fp32['provenance'] else 2
        fp32['files'] = {n:h for n,h in fp32['files'].items() if n in REQUIRED_FILES}
        first = _load_payloads(path,{**common,MANIFEST_NAME:_json_bytes(fp32),TENSORS_NAME:raw[TENSORS_NAME]},expected_model_fingerprint)
        second = _load_payloads(path,{**common,MANIFEST_NAME:raw[BF16_MANIFEST_NAME],TENSORS_NAME:raw[BF16_TENSORS_NAME]},expected_model_fingerprint)
        _validate_pair(first, second)
        selected = second if profile == 'fixed-voice-bf16' else first
        return VoicePack(path,selected.manifest,selected.tensors,tuple(DUAL_VARIANTS),manifest)

    actual_profile = manifest.get('provenance',{}).get('profile')
    if profile is not None and actual_profile is not None and profile != actual_profile:
        raise VoicePackError("音色包精度与运行时不匹配 (precision profile)")

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
    return VoicePack(path=path, manifest=manifest, tensors=tensors,
                     available_profiles=(actual_profile,) if actual_profile else ())


def _validate_pair(first, second):
    for field in ('voiceId','displayName','gender','language','indexTtsVersion','conditioningAbi','sourceModelFingerprint','license'):
        if first.manifest[field] != second.manifest[field]:
            raise VoicePackError(f"双精度音色元数据不一致: {field}")
    a,b = first.manifest.get('provenance',{}),second.manifest.get('provenance',{})
    if a.get('profile') != 'compatible-fp32' or b.get('profile') != 'fixed-voice-bf16':
        raise VoicePackError("必须使用分别预计算的 FP32、BF16 音色")
    for field in ('referenceSha256','preprocessFingerprint'):
        if a.get(field) != b.get(field):
            raise VoicePackError(f"双精度参考来源不一致: {field}")
    first_selection,second_selection=a.get('referenceSelection'),b.get('referenceSelection')
    # Revision tracks editor bookkeeping; compare every acoustic source field.
    if first_selection or second_selection:
        normalized=lambda s:{k:v for k,v in (s or {}).items() if k!='revision'}
        if normalized(first_selection)!=normalized(second_selection):
            raise VoicePackError("双精度参考来源不一致: referenceSelection")


def _write_payloads(destination, payloads, maximum=MAX_PACK_BYTES*2):
    destination=Path(destination).resolve(); destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',allowZip64=False,compresslevel=9) as archive:
            for name,data in sorted(payloads.items()):archive.writestr(_zip_info(name),data)
        if temporary.stat().st_size > maximum:raise VoicePackError("音色包超过大小限制")
        load_voicepack(temporary)
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def combine_voicepacks(output_path, fp32_path, bf16_path):
    """Bundle independently encoded profiles; preserve raw tensors and provenance."""
    first,second=load_voicepack(fp32_path),load_voicepack(bf16_path)
    if first.container_manifest or second.container_manifest:raise VoicePackError("合并输入必须为单精度包")
    _validate_pair(first,second)
    with zipfile.ZipFile(fp32_path) as a,zipfile.ZipFile(bf16_path) as b:
        payloads={n:a.read(n) for n in REQUIRED_FILES if n != MANIFEST_NAME}
        for name in (LICENSE_NAME,LICENSE_ZH_NAME,DISCLAIMER_NAME):
            if a.read(name)!=b.read(name):raise VoicePackError("双精度许可文件不一致")
        payloads[BF16_MANIFEST_NAME]=b.read(MANIFEST_NAME)
        payloads[BF16_TENSORS_NAME]=b.read(TENSORS_NAME)
    manifest=copy.deepcopy(first.manifest)
    manifest.update(schemaVersion=4,variants=DUAL_VARIANTS.copy(),files={n:_sha256(data) for n,data in payloads.items()})
    payloads[MANIFEST_NAME]=_json_bytes(manifest)
    return _write_payloads(output_path,payloads)


def extract_voicepack(source, destination, profile):
    """Export a legacy-compatible single profile without changing tensor bytes."""
    pack=load_voicepack(source,profile=profile)
    with zipfile.ZipFile(source) as archive:
        payloads={n:archive.read(n) for n in (LICENSE_NAME,LICENSE_ZH_NAME,DISCLAIMER_NAME)}
        tensor_name=BF16_TENSORS_NAME if pack.container_manifest and profile=='fixed-voice-bf16' else TENSORS_NAME
        payloads[TENSORS_NAME]=archive.read(tensor_name)
    payloads[MANIFEST_NAME]=_json_bytes(pack.manifest)
    return _write_payloads(destination,payloads,MAX_PACK_BYTES)


def repackage_voicepack(source, destination, name, gender):
    from .schema import validate_identity
    pack=load_voicepack(source); validate_identity(pack.voice_id,name,gender)
    with zipfile.ZipFile(source) as archive:payloads={n:archive.read(n) for n in archive.namelist()}
    manifest=copy.deepcopy(pack.container_manifest or pack.manifest)
    manifest.update(displayName=name,gender=gender)
    if pack.container_manifest:
        second=json.loads(payloads[BF16_MANIFEST_NAME])
        second.update(displayName=name,gender=gender)
        payloads[BF16_MANIFEST_NAME]=_json_bytes(second)
        manifest['files'][BF16_MANIFEST_NAME]=_sha256(payloads[BF16_MANIFEST_NAME])
    payloads[MANIFEST_NAME]=_json_bytes(manifest)
    return _write_payloads(destination,payloads)
