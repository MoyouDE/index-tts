import io
import json
import subprocess
import sys
import threading
import time

import pytest

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.runtime.emotion import (
    CALM_VECTOR,
    ExplicitEmotionProvider,
    QwenEmotionProvider,
    normalize_emotion,
)
from indextts.runtime.engine import SynthesisCancelled
from indextts.runtime.sidecar import JsonlSidecar


def test_slim_gpt_omits_reference_conditioners_and_text_head():
    model = UnifiedVoice(
        layers=1,
        model_dim=16,
        heads=2,
        max_text_tokens=12,
        max_mel_tokens=16,
        number_text_tokens=32,
        number_mel_codes=64,
        start_mel_token=62,
        stop_mel_token=63,
        checkpointing=False,
        condition_module=None,
        emo_condition_module=None,
        spk_cond_mode="campplus",
        precomputed_conditioning=True,
    )
    keys = set(model.state_dict())
    assert not any("conditioning_encoder" in key for key in keys)
    assert not any("perceiver" in key for key in keys)
    assert not any(key.startswith("spk_emb_proj") for key in keys)
    assert not any(key.startswith("text_head") for key in keys)
    assert "text_embedding.weight" in keys
    assert "mel_head.weight" in keys


def test_emotion_normalization_applies_bias_and_caps_total():
    vector = normalize_emotion([1.0] * 8)
    assert len(vector) == 8
    assert sum(vector) == pytest.approx(0.8)
    assert vector[2] > vector[6]
    assert ExplicitEmotionProvider([0, 0, 0, 0, 0, 0, 0, 0]).analyze("任意") == CALM_VECTOR


def test_qwen_failure_falls_back_to_calm_without_cuda(monkeypatch, tmp_path):
    provider = QwenEmotionProvider(tmp_path)

    def fail():
        raise RuntimeError("fake failure")

    monkeypatch.setattr(provider, "_load", fail)
    assert provider.analyze("这是一段文字") == CALM_VECTOR
    assert "回退 calm" in provider.warning


def test_qwen_invalid_json_is_rejected(tmp_path):
    provider = QwenEmotionProvider(tmp_path)
    with pytest.raises(ValueError, match="无效 JSON"):
        provider._decode_vector("not-json", "普通文本")


def test_runtime_import_does_not_load_reference_encoders_or_webui():
    script = r'''
import importlib
import json
import sys
importlib.import_module("indextts.runtime.engine")
forbidden = {
    "indextts.infer_v2_5",
    "indextts.s2mel.wav2vecbert_extract",
    "indextts.s2mel.modules.campplus.DTDNN",
    "indextts.utils.ja_g2p",
    "gradio",
    "pandas",
}
print(json.dumps(sorted(forbidden.intersection(sys.modules))))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(result.stdout) == []


class _FakeRuntime:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def health(self):
        print("model diagnostic must go to stderr")
        return {"status": "ready"}

    def list_voices(self):
        return [{"voiceId": "voice-1"}]

    def reload_voices(self):
        return self.list_voices()

    def synthesize(self, text, voice_id, emotion, duration_factor, _cancelled):
        self.started.set()
        while not self.release.wait(0.01):
            if _cancelled():
                raise SynthesisCancelled("fake cancelled")
        if _cancelled():
            raise SynthesisCancelled("fake cancelled")
        print("synthesis log must go to stderr")
        return {"audioPath": "cache/test.wav", "sampleRate": 22050, "durationMs": 1}


class _ProtocolInput:
    def __init__(self, runtime):
        self.runtime = runtime

    def __iter__(self):
        yield json.dumps({"id": "health", "method": "health", "params": {}}) + "\n"
        yield json.dumps({"id": "first", "method": "synthesize", "params": {"text": "甲", "voiceId": "voice-1"}}) + "\n"
        assert self.runtime.started.wait(2)
        yield json.dumps({"id": "second", "method": "synthesize", "params": {"text": "乙", "voiceId": "voice-1"}}) + "\n"
        yield json.dumps({"id": "cancel", "method": "cancel", "params": {"requestId": "second"}}) + "\n"
        self.runtime.release.set()
        time.sleep(0.05)
        yield json.dumps({"id": "bad", "method": "synthesize", "params": {"text": "丙", "voiceId": "voice-1", "outputPath": "C:/escape.wav"}}) + "\n"
        yield json.dumps({"id": "unknown", "method": "missing", "params": {}}) + "\n"
        yield json.dumps({"id": "stop", "method": "shutdown", "params": {}}) + "\n"


def test_jsonl_protocol_serializes_gpu_work_cancels_queue_and_keeps_stdout_clean():
    runtime = _FakeRuntime()
    stdout = io.StringIO()
    stderr = io.StringIO()
    server = JsonlSidecar(runtime, stdin=_ProtocolInput(runtime), stdout=stdout, stderr=stderr)
    assert server.serve() == 0

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    by_id = {response["id"]: response for response in responses}
    assert by_id["health"]["ok"] is True
    assert by_id["first"]["ok"] is True
    assert by_id["cancel"]["result"]["cancelled"] is True
    assert by_id["second"]["error"]["code"] == "cancelled"
    assert by_id["bad"]["error"]["code"] == "invalid_request"
    assert by_id["unknown"]["error"]["code"] == "method_not_found"
    assert all(set(response) >= {"id", "ok"} for response in responses)
    assert "diagnostic" not in stdout.getvalue()
    assert "stderr" in stderr.getvalue()
