"""Opt-in real-model isolation/cancellation/voice-switch regression.

INDEXTTS_TEST_MODEL_DIR=<export> pytest tests/test_reader_runtime_gpu.py
Default voice fixtures match the model profile. No source checkpoints are needed.
"""

import builtins
import importlib.abc
import io
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch

from indextts.runtime.engine import ReaderRuntime, SynthesisCancelled
from indextts.runtime.benchmark import tensor_digest
from indextts.runtime.profiles import BF16_RUNTIME_ABI

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("INDEXTTS_TEST_MODEL_DIR"), reason="Set INDEXTTS_TEST_MODEL_DIR for real-model tests")]


def test_reference_free_runtime_switch_cancel_and_recovery(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    model_dir = Path(os.environ["INDEXTTS_TEST_MODEL_DIR"]).resolve()
    manifest = json.loads((model_dir / "runtime_model.json").read_text(encoding="utf-8"))
    bf16 = manifest["runtimeAbi"] == BF16_RUNTIME_ABI
    voice_dir = root / "tests/fixtures" / ("voices-bf16" if bf16 else "voices-fp32")
    forbidden_paths = [root / "checkpoints", root / "examples"]
    forbidden_imports = ("indextts.infer_v2_5", "indextts.s2mel.wav2vecbert_extract",
                         "indextts.s2mel.modules.campplus", "onnxruntime", "gradio")

    class NoReferenceImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in forbidden_imports):
                raise AssertionError(f"Reference dependency imported: {fullname}")

    original_open = builtins.open
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file).resolve()
            if any(path.is_relative_to(forbidden) for forbidden in forbidden_paths):
                raise AssertionError(f"Reference file accessed: {path}")
        return original_open(file, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Reference-free inference must not access the network")

    finder = NoReferenceImports()
    sys.meta_path.insert(0, finder)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    try:
        torch.set_num_threads(4)
        runtime = ReaderRuntime(model_dir, [voice_dir], None, cache_dir=tmp_path)
        voices = [item["voiceId"] for item in runtime.list_voices()]
        assert len(voices) >= 2
        text = "\u6e05\u6668\u7684\u9633\u5149\u7167\u8fdb\u4e86\u5b89\u9759\u7684\u4e66\u623f\u3002"
        def generate(voice, emotion="base", cancelled=None):
            trace = {}
            result = runtime.synthesize(text, voice, emotion, seed=17,
                                        _cancelled=cancelled,
                                        _trace=lambda name, tensor: trace.update({name: tensor_digest(tensor)}))
            Path(result["audioPath"]).unlink()
            assert runtime.gpt.inference_model.cached_mel_emb is None
            return trace

        first = generate(voices[0])
        for _ in range(2):
            generate(voices[1], [0.2, 0, 0.3, 0, 0, 0, 0.2, 0.1])
            assert generate(voices[0]) == first
        assert generate(voices[0], [0.0] * 8) == first
        count = 0
        def cancelled():
            nonlocal count
            count += 1
            return count >= 4  # after GPT, before acoustic generation
        with pytest.raises(SynthesisCancelled):
            generate(voices[0], cancelled=cancelled)
        assert runtime.gpt.inference_model.cached_mel_emb is None
        assert not list(tmp_path.glob("*.part"))
        assert generate(voices[0]) == first
        with pytest.raises(KeyError):
            runtime.synthesize(text, "missing-voice", seed=17)
        assert generate(voices[0]) == first
        assert not any(name.startswith(forbidden_imports) for name in sys.modules)
    finally:
        sys.meta_path.remove(finder)
