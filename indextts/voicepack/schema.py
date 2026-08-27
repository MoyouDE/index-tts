"""Schema constants and validation for ``.ivp`` voice packs."""

from __future__ import annotations

import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
CONDITIONING_ABI = "indextts2.5-reader-v1"
MAX_PACK_BYTES = 10 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024

MANIFEST_NAME = "manifest.json"
TENSORS_NAME = "conditioning.safetensors"
LICENSE_NAME = "LICENSE"
LICENSE_ZH_NAME = "LICENSE_ZH.txt"
DISCLAIMER_NAME = "DERIVATIVE_DISCLAIMER.txt"

REQUIRED_FILES = {
    MANIFEST_NAME,
    TENSORS_NAME,
    LICENSE_NAME,
    LICENSE_ZH_NAME,
    DISCLAIMER_NAME,
}

TENSOR_RULES = {
    "speaker_latent": (2, {"bfloat16", "float16", "float32"}),
    "base_emotion": (2, {"bfloat16", "float16", "float32"}),
    "emotion_basis": (2, {"bfloat16", "float16", "float32"}),
    "prompt_condition": (3, {"bfloat16", "float16", "float32"}),
    "ref_mel": (3, {"float32"}),
    "speaker_style": (2, {"float32"}),
}

_VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VoicePackSchemaError(ValueError):
    """Raised when a voice-pack manifest or tensor set is invalid."""


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "voiceId",
        "displayName",
        "gender",
        "language",
        "indexTtsVersion",
        "conditioningAbi",
        "sourceModelFingerprint",
        "tensors",
        "files",
        "license",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise VoicePackSchemaError(f"manifest 缺少字段: {', '.join(missing)}")
    if set(manifest) - required:
        raise VoicePackSchemaError(f"manifest 含未知字段: {', '.join(sorted(set(manifest) - required))}")
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise VoicePackSchemaError(f"不支持的 schemaVersion: {manifest['schemaVersion']!r}")
    if not isinstance(manifest["voiceId"], str) or not _VOICE_ID.fullmatch(manifest["voiceId"]):
        raise VoicePackSchemaError("voiceId 必须为 1-64 位字母、数字、点、下划线或连字符")
    if not isinstance(manifest["displayName"], str) or not manifest["displayName"].strip():
        raise VoicePackSchemaError("displayName 不能为空")
    if len(manifest["displayName"]) > 128:
        raise VoicePackSchemaError("displayName 不能超过 128 个字符")
    if manifest["gender"] not in {"female", "male", "neutral", "unknown"}:
        raise VoicePackSchemaError("gender 必须为 female、male、neutral 或 unknown")
    if manifest["language"] != "zh":
        raise VoicePackSchemaError("schema v1 仅支持 language=zh")
    if manifest["conditioningAbi"] != CONDITIONING_ABI:
        raise VoicePackSchemaError(f"不支持的 conditioning ABI: {manifest['conditioningAbi']!r}")
    if not isinstance(manifest["sourceModelFingerprint"], str) or not _SHA256.fullmatch(
        manifest["sourceModelFingerprint"]
    ):
        raise VoicePackSchemaError("sourceModelFingerprint 必须为 SHA-256")

    tensors = manifest["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != set(TENSOR_RULES):
        raise VoicePackSchemaError("manifest 张量清单与 conditioning ABI 不匹配")
    for name, spec in tensors.items():
        if not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}:
            raise VoicePackSchemaError(f"张量 {name} 的描述无效")
        shape = spec["shape"]
        rank, dtypes = TENSOR_RULES[name]
        if not isinstance(shape, list) or len(shape) != rank or any(
            not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            raise VoicePackSchemaError(f"张量 {name} 的 shape 无效")
        if spec["dtype"] not in dtypes:
            raise VoicePackSchemaError(f"张量 {name} 的 dtype 无效: {spec['dtype']!r}")

    _validate_fixed_tensor_shapes(tensors)

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES - {MANIFEST_NAME}:
        raise VoicePackSchemaError("manifest 文件清单不完整")
    for name, digest in files.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise VoicePackSchemaError(f"文件 {name} 的 SHA-256 无效")

    license_info = manifest["license"]
    if not isinstance(license_info, dict) or license_info.get("model") != "bilibili Model Use License":
        raise VoicePackSchemaError("许可元数据无效")
    if license_info.get("voiceRights") != "provided-separately":
        raise VoicePackSchemaError("音色权利声明缺失")


def _validate_fixed_tensor_shapes(tensors: Mapping[str, Any]) -> None:
    expected = {
        "speaker_latent": (1, 1280),
        "base_emotion": (1, 1280),
        "emotion_basis": (8, 1280),
        "speaker_style": (1, 192),
    }
    for name, shape in expected.items():
        if tuple(tensors[name]["shape"]) != shape:
            raise VoicePackSchemaError(f"张量 {name} 维度不符合 ABI: {tensors[name]['shape']}")
    if tensors["prompt_condition"]["shape"][0] != 1 or tensors["prompt_condition"]["shape"][2] != 512:
        raise VoicePackSchemaError("prompt_condition 维度不符合 ABI")
    if tensors["ref_mel"]["shape"][0] != 1 or tensors["ref_mel"]["shape"][1] != 80:
        raise VoicePackSchemaError("ref_mel 维度不符合 ABI")
