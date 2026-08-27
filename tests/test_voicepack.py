import json
import zipfile
from pathlib import Path

import pytest
import torch

from indextts.voicepack.archive import VoicePackError, load_voicepack, write_voicepack
from indextts.voicepack.builder import VoicePackBuilder
from indextts.voicepack.schema import (
    CONDITIONING_ABI,
    DISCLAIMER_NAME,
    LICENSE_NAME,
    LICENSE_ZH_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    TENSORS_NAME,
)


def _tensors():
    return {
        "speaker_latent": torch.zeros(1, 1280, dtype=torch.bfloat16),
        "base_emotion": torch.zeros(1, 1280, dtype=torch.bfloat16),
        "emotion_basis": torch.zeros(8, 1280, dtype=torch.float32),
        "prompt_condition": torch.zeros(1, 24, 512, dtype=torch.float32),
        "ref_mel": torch.zeros(1, 80, 20, dtype=torch.float32),
        "speaker_style": torch.zeros(1, 192, dtype=torch.float32),
    }


def _manifest():
    return {
        "schemaVersion": SCHEMA_VERSION,
        "voiceId": "reader-female-01",
        "displayName": "阅读女声",
        "gender": "female",
        "language": "zh",
        "indexTtsVersion": "2.5",
        "conditioningAbi": CONDITIONING_ABI,
        "sourceModelFingerprint": "a" * 64,
        "tensors": {},
        "files": {},
        "license": {
            "model": "bilibili Model Use License",
            "modelLicenseFile": LICENSE_NAME,
            "derivativeDisclaimerFile": DISCLAIMER_NAME,
            "voiceRights": "provided-separately",
        },
    }


def _licenses():
    return {
        LICENSE_NAME: b"license",
        LICENSE_ZH_NAME: b"license zh",
        DISCLAIMER_NAME: b"derived work",
    }


def test_voicepack_is_deterministic_and_pickle_free(tmp_path):
    first = write_voicepack(tmp_path / "first.ivp", _manifest(), _tensors(), _licenses())
    second = write_voicepack(tmp_path / "second.ivp", _manifest(), _tensors(), _licenses())

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 10 * 1024 * 1024
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert TENSORS_NAME in names
        assert not any(name.lower().endswith((".wav", ".pkl", ".pickle", ".pt", ".pth")) for name in names)
        assert b"E:\\" not in archive.read(MANIFEST_NAME)

    pack = load_voicepack(first, expected_model_fingerprint="a" * 64)
    assert pack.voice_id == "reader-female-01"
    assert set(pack.tensors) == set(_tensors())


def test_voicepack_rejects_corrupt_hash(tmp_path):
    path = write_voicepack(tmp_path / "voice.ivp", _manifest(), _tensors(), _licenses())
    with zipfile.ZipFile(path, "r") as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads[TENSORS_NAME] += b"corrupt"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)

    with pytest.raises(VoicePackError, match="哈希不匹配"):
        load_voicepack(path)


def test_voicepack_rejects_path_traversal_before_extracting(tmp_path):
    path = tmp_path / "traversal.ivp"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../manifest.json", b"{}")
    with pytest.raises(VoicePackError, match="不安全路径"):
        load_voicepack(path)
    assert not (tmp_path.parent / "manifest.json").exists()


def test_voicepack_rejects_model_fingerprint_mismatch(tmp_path):
    path = write_voicepack(tmp_path / "voice.ivp", _manifest(), _tensors(), _licenses())
    with pytest.raises(VoicePackError, match="模型指纹"):
        load_voicepack(path, expected_model_fingerprint="b" * 64)


def test_builder_uses_precomputed_conditioning_without_storing_reference(tmp_path):
    reference = tmp_path / "private-source.wav"
    reference.write_bytes(b"not decoded by the fake model")

    class FakeTTS:
        model_version = "2.5-test"

        def extract_voice_conditioning(self, audio_path, verbose=False):
            assert audio_path == str(reference.resolve())
            return _tensors()

    builder = VoicePackBuilder(
        FakeTTS(), model_dir=tmp_path, source_model_fingerprint="c" * 64
    )
    output = builder.build(
        reference,
        {"voiceId": "novel-01", "displayName": "小说女声", "gender": "female"},
        tmp_path / "novel.ivp",
    )
    with zipfile.ZipFile(output) as archive:
        manifest_text = archive.read(MANIFEST_NAME).decode("utf-8")
        assert str(reference) not in manifest_text
        assert reference.name not in manifest_text
        manifest = json.loads(manifest_text)
        assert manifest["voiceId"] == "novel-01"
