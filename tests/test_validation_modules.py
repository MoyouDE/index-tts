import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import soundfile as sf
import torch

from indextts.runtime.emotion import OnnxEmotionProvider
from indextts.validation_service import ValidationService, audio_details, context_rows
from indextts.voicepack.builder import VoicePackBuilder
from indextts.voicepack.reference import ReferenceEncoder, load_reference_audio


@pytest.mark.parametrize("rate,channels,seconds", [(16000, 1, 1), (48000, 2, 1), (22050, 1, 16)])
def test_reference_preprocessing_keeps_historical_samples(tmp_path, rate, channels, seconds):
    import librosa
    audio = np.sin(np.arange(rate * seconds) * 2 * np.pi * 220 / rate).astype(np.float32) * 0.2
    if channels == 2:
        audio = np.stack([audio, audio * 0.5], axis=1)
    path = tmp_path / "source.wav"
    sf.write(path, audio, rate)
    details = audio_details(path)
    assert details["channels"] == channels
    assert details["truncated"] == (seconds > 15)
    actual, actual_rate = load_reference_audio(path)
    expected, expected_rate = librosa.load(path)
    assert actual_rate == expected_rate == 22050
    assert torch.equal(actual, torch.tensor(expected).unsqueeze(0)[:, :15 * 22050])


def test_bad_audio_and_metadata_do_not_initialize_models(tmp_path, monkeypatch):
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"broken")
    builder = VoicePackBuilder(model_dir=tmp_path)
    monkeypatch.setattr(builder, "_get_tts", lambda: pytest.fail("Model loaded for invalid identity"))
    with pytest.raises(ValueError, match="voiceId"):
        builder.build(audio, {"voiceId": "../bad", "displayName": "bad"}, tmp_path / "out.ivp")
    with pytest.raises(Exception):
        audio_details(audio)


def test_explicit_bf16_cpu_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text("version: '2.5'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BF16"):
        ReferenceEncoder(model_dir=tmp_path, device="cpu", use_bf16=True)


def test_sessions_cannot_import_other_session_results(tmp_path):
    service = ValidationService(tmp_path)
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    original = service.session_dir(a) / "private.ivp"
    original.write_bytes(b"private")
    with pytest.raises(ValueError, match="其他会话"):
        service.inspect(original, "", b)
    with pytest.raises(ValueError, match="会话"):
        service.session_dir("../escape")


def test_context_editor_preserves_boundaries_and_validates_paragraphs():
    result = context_rows([["s1", "a", 0, "对白", "你好"], ["s2", "b", 1, "旁白", "再见"]])
    assert result[0]["sentenceType"] == "dialogue"
    assert result[1]["sectionId"] == "b"
    for line in (-1, 1.5, "nan", True):
        with pytest.raises(ValueError, match="段落"):
            context_rows([["s", "a", line, "对白", "text"]])


def fake_provider(tmp_path, vector=None, intensity=0.3):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                return {"input_ids": list(range(len(text)))}
            return {"input_ids": np.ones((1, 8), dtype=np.int64), "attention_mask": np.ones((1, 8), dtype=np.int64)}
    class Session:
        calls = 0
        def run(self, outputs, inputs):
            self.calls += 1
            return np.array([vector if vector is not None else [0.2] * 8]), np.array([[intensity]])
    provider = OnnxEmotionProvider(tmp_path, neutral_threshold=0.2)
    provider._session, provider._tokenizer = Session(), Tokenizer()
    return provider


def sentences():
    return context_rows([["a", "chapter", 0, "旁白", "他回来了。"],
                         ["b", "chapter", 0, "对白", "太好了！"]])


def test_emotion_detail_single_inference_and_legacy_vector(tmp_path):
    provider = fake_provider(tmp_path)
    result = provider.analyze_window_details(sentences(), "b")
    assert provider._session.calls == 1
    assert result["rawVector"] == [0.2] * 8
    assert result["vector"] == provider.analyze_window(sentences(), "b")
    assert "[TGT][对白]太好了！[/TGT]" in result["context"]["rendered_text"]
    base = provider.analyze_window_details(sentences(), "b", neutral_threshold=0.4)
    assert base["baseFallback"] and base["vector"] == [0.0] * 8
    assert base["rawVector"] == result["rawVector"]
    assert provider.neutral_threshold == 0.2


def test_emotion_context_rejects_cross_section(tmp_path):
    rows = sentences()
    rows[0]["sectionId"] = "other"
    with pytest.raises(ValueError, match="section"):
        fake_provider(tmp_path).analyze_window_details(rows, "b")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_threshold_validation(tmp_path, bad):
    with pytest.raises(ValueError, match="threshold"):
        fake_provider(tmp_path).analyze_window_details(sentences(), "b", neutral_threshold=bad)


@pytest.mark.parametrize("vector,intensity", [([float("nan")] * 8, 0.3), ([0.0] * 8, float("inf")), ([0.0] * 7, 0.3)])
def test_bad_onnx_outputs_raise(tmp_path, vector, intensity):
    with pytest.raises(ValueError, match="输出"):
        fake_provider(tmp_path, vector, intensity).analyze_window_details(sentences(), "b")


def test_onnx_model_invalid_manifest_raises(tmp_path):
    pytest.importorskip("onnxruntime")
    (tmp_path / "emotion_model.json").write_text(json.dumps({"conditioningAbi": "broken"}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        OnnxEmotionProvider(tmp_path)._load()


def test_producer_failure_can_recover_and_switch_releases(tmp_path, monkeypatch):
    import indextts.voicepack.builder as module
    from test_voicepack import _manifest, _tensors, _licenses
    from indextts.voicepack.archive import write_voicepack
    roots = tmp_path / "models"
    roots.mkdir()
    (roots / "config.yaml").write_text("{}")
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000), 16000)
    created = []
    class FakeBuilder:
        def __init__(self, **kwargs):
            self.closed = False
            created.append(self)
        def close(self):
            self.closed = True
        def source_model_fingerprint(self):
            return "a" * 64
        def build(self, reference, metadata, destination):
            if len(created) == 1:
                raise RuntimeError("simulated GPU failure")
            return write_voicepack(destination, _manifest(), _tensors(), _licenses())
    monkeypatch.setattr(module, "VoicePackBuilder", FakeBuilder)
    service = ValidationService(tmp_path / "outputs")
    args = (str(audio), "voice", "Voice", "unknown", "compatible-fp32", "auto", str(roots), uuid.uuid4().hex)
    with pytest.raises(RuntimeError, match="GPU"):
        service.build(*args)
    assert created[0].closed
    first, report = service.build(*args)
    second, _ = service.build(*args)
    assert first != second and Path(first).is_file() and report["ok"]
    service.unload_producer()
    assert created[-1].closed


def test_invalid_directories_and_corrupt_pack_do_not_poison_service(tmp_path):
    from test_voicepack import _manifest, _tensors, _licenses
    from indextts.voicepack.archive import write_voicepack
    service = ValidationService(tmp_path / "outputs")
    session = uuid.uuid4().hex
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="config.yaml"):
        service.build(audio, "voice", "Voice", "unknown", "compatible-fp32", "auto", tmp_path, session)
    with pytest.raises(ValueError, match="情感模型目录"):
        service.emotion_default("")
    broken = tmp_path / "bad.ivp"
    broken.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        service.inspect(broken, "", session)
    valid = write_voicepack(tmp_path / "good.ivp", _manifest(), _tensors(), _licenses())
    assert service.inspect(valid, "", session)["ok"]
    # Invalid requests remain recoverable through the public compatibility API.
    assert service.unload_producer() == "制包模型已卸载"
    assert service.unload_emotion() == "情感模型已卸载"
