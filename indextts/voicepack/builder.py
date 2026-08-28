"""Build reusable voice conditioning packs from one reference recording."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from .archive import write_voicepack
from .schema import CONDITIONING_ABI, DISCLAIMER_NAME, LICENSE_NAME, LICENSE_ZH_NAME, SCHEMA_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(model_dir: str | Path, cfg_path: str | Path | None = None) -> str:
    """Hash all runtime-relevant source weights into one stable fingerprint."""
    model_root = Path(model_dir).resolve()
    config_path = Path(cfg_path).resolve() if cfg_path else model_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少模型配置: {config_path}")
    cfg = OmegaConf.load(config_path)
    candidates = [
        config_path,
        model_root / str(cfg.gpt_checkpoint),
        model_root / str(cfg.s2mel_checkpoint),
        model_root / "codec.pth",
        model_root / "hf_cache" / "bigvgan" / str(cfg.vocoder.name),
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"计算模型指纹时缺少文件: {path}")
        relative = path.relative_to(model_root) if path.is_relative_to(model_root) else Path(path.name)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class VoicePackBuilder:
    """Developer-side builder. The full model is loaded only when required."""

    def __init__(
        self,
        tts: Any | None = None,
        *,
        model_dir: str | Path = "checkpoints",
        cfg_path: str | Path | None = None,
        device: str | None = None,
        use_bf16: bool = False,
        source_model_fingerprint: str | None = None,
    ) -> None:
        self._tts = tts
        self.model_dir = Path(model_dir).resolve()
        self.cfg_path = Path(cfg_path).resolve() if cfg_path else self.model_dir / "config.yaml"
        self.device = device
        self.use_bf16 = use_bf16
        self._source_model_fingerprint = source_model_fingerprint

    def _get_tts(self):
        if self._tts is None:
            from indextts.infer_v2_5 import IndexTTS2

            self._tts = IndexTTS2(
                cfg_path=str(self.cfg_path),
                model_dir=str(self.model_dir),
                device=self.device,
                use_bf16=self.use_bf16,
                use_qwen_emo=False,
            )
        return self._tts

    def source_model_fingerprint(self) -> str:
        """Return and cache the fingerprint used to validate compatible packs."""
        if self._source_model_fingerprint is None:
            self._source_model_fingerprint = model_fingerprint(self.model_dir, self.cfg_path)
        return self._source_model_fingerprint

    def build(
        self,
        reference_audio: str | Path,
        metadata: Mapping[str, Any],
        output_path: str | Path,
    ) -> Path:
        reference = Path(reference_audio).resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"参考音频不存在: {reference}")
        voice_id = str(metadata.get("voiceId", "")).strip()
        display_name = str(metadata.get("displayName", metadata.get("name", ""))).strip()
        gender = str(metadata.get("gender", "unknown")).strip().lower()

        tts = self._get_tts()
        tensors = tts.extract_voice_conditioning(str(reference), verbose=False)
        fingerprint = self.source_model_fingerprint()
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "voiceId": voice_id,
            "displayName": display_name,
            "gender": gender,
            "language": "zh",
            "indexTtsVersion": str(getattr(tts, "model_version", "2.5")),
            "conditioningAbi": CONDITIONING_ABI,
            "sourceModelFingerprint": fingerprint,
            "tensors": {},
            "files": {},
            "license": {
                "model": "bilibili Model Use License",
                "modelLicenseFile": LICENSE_NAME,
                "derivativeDisclaimerFile": DISCLAIMER_NAME,
                "voiceRights": "provided-separately",
            },
        }
        root = Path(__file__).resolve().parents[2]
        upstream_disclaimer = (root / "DISCLAIMER").read_text(encoding="utf-8")
        derivative = (
            upstream_disclaimer.rstrip()
            + "\n\nDERIVATIVE WORK NOTICE\n"
            + "This voice pack is a derived conditioning artifact produced from IndexTTS model outputs. "
            + "It is not an official IndexTTS model release. Rights and consent for the reference voice "
            + "must be obtained separately by the voice provider.\n"
        ).encode("utf-8")
        licenses = {
            LICENSE_NAME: (root / "LICENSE").read_bytes(),
            LICENSE_ZH_NAME: (root / "LICENSE_ZH.txt").read_bytes(),
            DISCLAIMER_NAME: derivative,
        }
        return write_voicepack(output_path, manifest, tensors, licenses)


def describe_manifest(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
