import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

import pytest

from indextts.web_modules import parse_modules
from indextts.gpu_coordinator import GpuCoordinator
from indextts.validation_web import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("invalid", ["", "producer,", "other", "emotion,emotion", [], [None]])
def test_invalid_module_selection_rejected_before_page_construction(invalid):
    with pytest.raises(ValueError):
        create_app(modules=invalid)


def test_selection_uses_fixed_page_order():
    assert parse_modules("audition, producer") == ("producer", "audition")
    assert parse_modules() == ("producer", "emotion", "audition")


@pytest.mark.parametrize("mode", ["producer,emotion,audition", "producer", "emotion", "audition", "producer,audition", "emotion,audition"])
def test_fresh_process_feature_isolation_and_no_model_loading(mode):
    run = subprocess.run([sys.executable, "tools/measure_validation_startup.py", "--mode", mode],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180, check=True)
    data = json.loads(run.stdout.strip().splitlines()[-1])
    enabled = parse_modules(mode)
    assert data["tabs"] == [{"producer": "音色包生成", "emotion": "情感推理", "audition": "合成试听"}[m] for m in enabled]
    assert data["workspaceCreated"] == (enabled != ("emotion",))
    loaded = data["loadedModules"]
    for feature in ("producer", "emotion", "audition"):
        assert loaded[f"indextts.{feature}_web"] == (feature in enabled)
        if feature not in enabled:
            assert not loaded[f"indextts.{feature}_service"]
    assert not loaded["indextts.voicepack.reference"]
    assert not loaded["indextts.runtime.engine"]
    assert not loaded["onnxruntime"]
    assert not loaded["indextts.voicepack.builder"]
    if enabled == ("emotion",):
        assert not loaded["indextts.voicepack.archive"]
        assert not loaded["torch"]


def test_cross_page_options_only_when_targets_enabled(tmp_path):
    for mode in ("audition", "producer,audition", "emotion,audition", None):
        app = create_app(modules=mode, workspace_dir=tmp_path/"work", output_dir=tmp_path/"out")
        components = app.config["components"]
        emotion = next(c["props"] for c in components if c["props"].get("label") == "试听情感")
        assert ("采用情感页结果" in [v[0] for v in emotion["choices"]]) == (mode is None or "emotion" in mode)
        missing = next(c["props"] for c in components if c["props"].get("value") == "前往制包页生成所选精度")
        assert missing["visible"] == (mode is None or "producer" in mode)
        app.workbench.close()


def test_entrypoint_owns_missing_precision_navigation(tmp_path):
    app = create_app(modules="producer,audition", workspace_dir=tmp_path/"work", output_dir=tmp_path/"out")
    button = next(c["id"] for c in app.config["components"]
                  if c["props"].get("value") == "前往制包页生成所选精度")
    dependency = next(d for d in app.config["dependencies"]
                      if any(tuple(target) == (button, "click") for target in d["targets"]))
    updates = app.fns[dependency["id"]].fn(None, "fixed-voice-bf16")
    assert updates[0]["selected"] == "voices"
    assert updates[2] == "fixed-voice-bf16"
    app.workbench.close()


def test_audition_page_copies_bridge_vector_without_calling_emotion(tmp_path, monkeypatch):
    app = create_app(modules="emotion,audition", workspace_dir=tmp_path/"work", output_dir=tmp_path/"out")
    received = []
    def synthesize(*args):
        received.append(args[5])
        return None, None, {"modelLoadMs": 0, "seed": 17,
            "result": {"timings": {"totalMs": 100}, "durationMs": 1000, "rtf": .1}}
    monkeypatch.setattr(app.workbench, "audition", synthesize)
    monkeypatch.setattr(app.workbench, "analyze", lambda *args: pytest.fail("Audition called emotion service"))
    fn = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "synth")
    vector = [.2]*8
    fn("voice", "compatible-fp32", "model", "cuda:0", "text", "采用情感页结果", "[]", vector,
        1, False, 17, "全部启用", [], 4, "native", uuid.uuid4().hex,
        False, .8, .8, 30, 3, 10, 0, 1500, 25, .7)
    assert received == [vector] and received[0] is not vector
    vector[0] = .9
    assert received[0][0] == .2
    app.workbench.close()


def test_gpu_coordinator_serializes_and_releases_other_owner():
    gpu = GpuCoordinator()
    events = []
    gpu.bind("producer", lambda: events.append("release-producer"))
    gpu.bind("audition", lambda: events.append("release-audition"))
    entered = threading.Event()
    def audition():
        entered.set()
        with gpu.use("audition"):
            events.append("audition")
    with gpu.use("producer"):
        events.append("producer")
        thread = threading.Thread(target=audition)
        thread.start()
        assert entered.wait(2)
        assert events == ["release-audition", "producer"]
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert events == ["release-audition", "producer", "release-producer", "audition"]
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="试听已取消"):
        with gpu.use("audition", cancelled):
            pytest.fail("Cancelled request entered GPU lane")
    assert events == ["release-audition", "producer", "release-producer", "audition"]


def test_facade_composition_and_disabled_operations(tmp_path):
    from indextts.validation_service import ValidationService
    from indextts.workbench_service import WorkbenchService
    assert not issubclass(WorkbenchService, ValidationService)
    service = WorkbenchService(tmp_path/"out", tmp_path/"work", modules="emotion")
    with pytest.raises(ValueError, match="未启用"):
        service.build(None, "voice", "Voice", "unknown", "compatible-fp32", "auto", "missing", uuid.uuid4().hex)
    with pytest.raises(ValueError, match="未启用"):
        service.audition_service
    assert not (tmp_path/"work").exists()
    service.close()


def test_emotion_bridge_keeps_vector_copy_and_failed_analysis_preserves_state(tmp_path, monkeypatch):
    app = create_app(modules="emotion,audition", workspace_dir=tmp_path/"work", output_dir=tmp_path/"out")
    result = {"rawVector": [.2]*8, "vector": [.1]*8, "totalIntensity": .8,
              "threshold": .2, "baseFallback": False, "context": {"rendered_text": "target"}}
    monkeypatch.setattr(app.workbench, "analyze", lambda *args: (result, None))
    fn = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "analyze")
    first = fn()
    assert first[-1] == result["vector"] and first[-1] is not result["vector"]
    result["vector"][0] = .9
    assert first[-1][0] == .1
    def fail(*args):
        raise ValueError("bad model")
    monkeypatch.setattr(app.workbench, "analyze", fail)
    with pytest.raises(Exception, match="bad model"):
        fn()
    assert first[-1][0] == .1
    app.workbench.close()
