import json
from pathlib import Path
import subprocess
import runpy
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


def test_saved_material_is_not_implicitly_selected_on_page_construction(tmp_path):
    from indextts.material_service import MaterialService,write_json
    sid=uuid.uuid4().hex
    directory=MaterialService(tmp_path/"work").directory(sid);directory.mkdir()
    write_json(directory/"material.json",{"sourceId":sid,"name":"reference.mp4"})
    app=create_app(modules="producer",workspace_dir=tmp_path/"work",output_dir=tmp_path/"out")
    try:
        dropdown=next(c for c in app.config["components"] if c["props"].get("label")=="已保存素材")
        assert dropdown["props"].get("value") is None
        assert dropdown["props"]["choices"]==[(f"reference.mp4 · {sid[:8]}",sid)]
    finally:app.workbench.close()


@pytest.mark.parametrize("custom_directory", [False, True])
def test_producer_directory_is_fixed_on_server(tmp_path, monkeypatch, custom_directory):
    from indextts.workbench_service import WorkbenchService
    calls = []
    def capture(operation):
        def invoke(self, *args):
            calls.append((operation, args))
            return ("voice.ivp", {"seconds": 1}) if operation != "inspect" else {}
        return invoke
    for operation in ("build_selection", "rebuild", "inspect"):
        monkeypatch.setattr(WorkbenchService, operation, capture(operation))
    monkeypatch.chdir(tmp_path)
    options = {"source_model_dir": "fixed-models"} if custom_directory else {}
    expected = str(tmp_path / "fixed-models" if custom_directory else ROOT / "voice-producer" / "models" / "checkpoints")
    app = create_app(modules="producer", workspace_dir=tmp_path/"work", output_dir=tmp_path/"out", **options)
    try:
        source = next(c for c in app.config["components"] if c["props"].get("label") == "源模型目录")
        assert source["props"]["interactive"] is False
        assert source["props"]["value"] == expected
        assert all(source["id"] not in d["inputs"] for d in app.config["dependencies"])
        # Even a changed display value cannot redirect the server's operations.
        app.blocks[source["id"]].value = "other-models"
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        sid = uuid.uuid4().hex
        callbacks["generate_snapshot"]({"selection": {"method": "primary-only-v1"},
            "args": ["Voice", "unknown", "compatible-fp32", "auto", sid]})
        callbacks["regenerate"]("voice", "fixed-voice-bf16", "auto", sid)
        callbacks["inspect"]("voice.ivp", sid)
        assert [(name, args[{"build_selection": 6, "rebuild": 3, "inspect": 1}[name]]) for name, args in calls] == [
            ("build_selection", expected), ("rebuild", expected), ("inspect", expected)]
    finally:
        app.workbench.close()


def test_standalone_producer_entry_uses_fixed_directories_from_other_cwd(tmp_path, monkeypatch):
    import indextts.validation_web as web
    received = []
    monkeypatch.setattr(web, "main", lambda argv: received.append(argv))
    monkeypatch.chdir(tmp_path)
    entry = runpy.run_path(str(ROOT / "voice-producer/start.py"))
    entry["main"](["--port", "7863"])
    options = dict(zip(received[0][::2], received[0][1::2]))
    assert options["--modules"] == "producer"
    assert options["--source-model-dir"] == web.DEFAULT_SOURCE_MODEL_DIR
    assert options["--workspace-dir"] == str(ROOT / "outputs/voice-workbench")
    assert options["--port"] == "7863"


def test_producer_asset_lock_reuses_existing_hashes_without_unrelated_assets():
    lock = json.loads((ROOT / "voice-producer/assets.lock.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / "tests/fixtures/reader-assets.lock.json").read_text(encoding="utf-8"))
    expected = set(original["files"]) - {"examples/voice_01.wav", "examples/voice_02.wav",
        "checkpoints/multilingual_zh_ja_yue_char_del.tiktoken", "checkpoints/hf_cache/bigvgan/config.json"}
    assert set(lock["files"]) == expected | {"vad/silero-v6.0.onnx"}
    assert lock["schemaVersion"] == 1
    assert all(lock["files"][name] == original["files"][name] for name in expected)
    assert "/v6.0/" in lock["files"]["vad/silero-v6.0.onnx"]["url"]


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
    assert loaded["indextts.material_web"] == ("producer" in enabled)
    assert loaded["indextts.material_service"] == ("producer" in enabled)
    assert not loaded["indextts.material_vad"]
    if enabled == ("emotion",):
        assert not loaded["indextts.voicepack.archive"]
        assert not loaded["torch"]


def test_cross_page_options_only_when_targets_enabled(tmp_path):
    for mode in ("audition", "producer,audition", "emotion,audition", None):
        app = create_app(modules=mode, workspace_dir=tmp_path/"work", output_dir=tmp_path/"out")
        components = app.config["components"]
        types = {c["id"]: c["type"] for c in components}
        def check_tabs(node):
            if types.get(node["id"]) == "tabs":
                # A State directly inside Tabs becomes an extra visible tab in Gradio.
                assert all(types[child["id"]] == "tabitem" for child in node.get("children", []))
            for child in node.get("children", []):
                check_tabs(child)
        check_tabs(app.config["layout"])
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
