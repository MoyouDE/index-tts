import copy
import json
from pathlib import Path
import threading
import uuid

import pytest
import torch

from indextts.runtime.profiles import FP32, BF16, generation_options, synthesis_settings
from indextts.runtime.engine import ReaderRuntime
from indextts.voice_workspace import VoiceWorkspace
from indextts.voicepack.archive import load_voicepack, write_voicepack
from indextts.voicepack.provenance import PREPROCESS_FINGERPRINT, sha256_file
from indextts.workbench_service import WorkbenchService
from test_voicepack import _manifest, _tensors, _licenses


def pack(path, profile=FP32, reference_hash="b"*64, value=0):
    manifest = _manifest()
    manifest.update(schemaVersion=2, provenance={"profile": profile, "referenceSha256": reference_hash,
        "referenceEncoderFingerprint": "c"*64, "preprocessFingerprint": PREPROCESS_FINGERPRINT,
        "producerVersions": {"torch": torch.__version__}})
    tensors = _tensors()
    if profile == FP32:
        tensors = {name: tensor.float() for name, tensor in tensors.items()}
    tensors["base_emotion"].fill_(value)
    return write_voicepack(path, manifest, tensors, _licenses())


def test_library_two_profiles_atomic_replacement_restart_and_delete(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixed original recording")
    source = pack(tmp_path / "first.ivp", reference_hash=sha256_file(reference))
    store = VoiceWorkspace(tmp_path / "work")
    dest = store.install(source, reference=reference)
    vid = load_voicepack(source).voice_id
    instance = store.record(vid)["instance"]
    second = pack(tmp_path / "second.ivp", BF16, sha256_file(reference))
    store.install(second, expected_instance=instance)
    assert all(v["ready"] for v in VoiceWorkspace(store.root).items()[0]["variants"].values())
    store.notes(vid, "喜欢这个声音")
    assert VoiceWorkspace(store.root).record(vid)["notes"] == "喜欢这个声音"
    before = Path(dest).read_bytes()
    corrupt = tmp_path / "corrupt.ivp"
    corrupt.write_bytes(b"broken")
    with pytest.raises(Exception):
        store.install(corrupt, expected_instance=instance)
    assert Path(dest).read_bytes() == before
    changed = pack(tmp_path / "changed.ivp", reference_hash="d"*64)
    with pytest.raises(ValueError, match="参考音频"):
        store.install(changed, expected_instance=instance)
    with pytest.raises(ValueError, match="确认"):
        store.delete(vid, "wrong")
    store.delete(vid, f"{vid}:{instance}")
    with pytest.raises(ValueError, match="已删除"):
        store.install(source, expected_instance=instance)
    assert store.items() == []


def test_latest_preview_failure_preserves_previous_and_snapshot_survives_replace(tmp_path):
    source = pack(tmp_path / "first.ivp")
    store = VoiceWorkspace(tmp_path / "work")
    store.install(source)
    vid = load_voicepack(source).voice_id
    snapshot = tmp_path / "snapshot.ivp"
    instance, digest = store.snapshot(vid, FP32, snapshot)
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"WAV data")
    store.save_preview(vid, FP32, instance, digest, audio, {"seed": 17})
    old = store.latest(vid, FP32)
    with pytest.raises(FileNotFoundError):
        store.save_preview(vid, FP32, instance, digest, tmp_path/"missing.wav", {})
    assert store.latest(vid, FP32) == old
    store.save_preview(vid, FP32, instance, digest, audio, {"seed": 18})
    directory = Path(store.latest(vid, FP32)[0]).parent
    assert len(list(directory.glob("*.wav"))) == 1
    assert len(list(directory.glob("*.json"))) == 2
    assert sha256_file(snapshot) == digest


def test_import_cannot_overwrite_and_has_no_reference(tmp_path):
    source = pack(tmp_path / "first.ivp")
    store = VoiceWorkspace(tmp_path / "work")
    store.install(source)
    assert store.items()[0]["reference"] is None
    with pytest.raises(ValueError, match="已存在"):
        store.install(source)
    service = WorkbenchService(tmp_path/"out", store.root)
    with pytest.raises(ValueError, match="没有参考音频"):
        service.rebuild(load_voicepack(source).voice_id, BF16, "auto", "unused", uuid.uuid4().hex)
    service.close()


@pytest.mark.parametrize("profile", [FP32, BF16])
def test_defaults_keep_exact_generation_options(profile):
    values = synthesis_settings(profile)
    assert {k:v for k,v in values.items() if k not in {"cfg", "acoustic_steps"}} == generation_options(profile)
    assert values["cfg"] == .7 and values["acoustic_steps"] == 25


@pytest.mark.parametrize("overrides", [{"unknown": 1}, {"top_k": 1.5}, {"temperature": float("nan")},
    {"cfg": 3.1}, {"acoustic_steps": 0}, {"num_beams": True}, {"do_sample": 1}, {"max_generate_length": 1600}])
def test_reject_bad_settings(overrides):
    with pytest.raises(ValueError):
        synthesis_settings(FP32, overrides, max_tokens=1500)


def test_runtime_load_rejects_mixed_precision_and_snapshots_keep_old_pack(tmp_path):
    runtime = ReaderRuntime.__new__(ReaderRuntime)
    runtime.profile, runtime.source_fingerprint = FP32, "a"*64
    runtime._voice_lock = threading.Lock()
    first = pack(tmp_path/"first.ivp")
    second = pack(tmp_path/"second.ivp", value=1)
    bad = pack(tmp_path/"bad.ivp", BF16)
    vid = runtime.load_voice(first)["voiceId"]
    snapshot = runtime.snapshot_voice(vid)
    runtime.load_voice(second)
    assert not torch.equal(snapshot.tensors["base_emotion"], runtime.snapshot_voice(vid).tensors["base_emotion"])
    with pytest.raises(ValueError, match="precision"):
        runtime.load_voice(bad)


def test_audition_snapshots_parameters_saves_latest_and_recovers_worker_failure(tmp_path, monkeypatch):
    import indextts.audition_service as module
    source = pack(tmp_path/"voice.ivp")
    service = WorkbenchService(tmp_path/"out", tmp_path/"work")
    service.library.install(source)
    vid = load_voicepack(source).voice_id
    model = tmp_path/"model"
    model.mkdir()
    (model/"runtime_model.json").write_text(json.dumps({"runtimeAbi": "indextts2.5-reader-runtime-v2", "sourceModelFingerprint":"a"*64}))
    instances = []
    class Worker:
        def __init__(self, *args):
            self.process = self
            self.closed = False
            self.failed = False
            instances.append(self)
        def poll(self): return 0 if self.closed else None
        def close(self): self.closed = True
        def call(self, method, params=None, **kwargs):
            if method == "synthesize":
                if self.failed:
                    raise RuntimeError("simulated failure")
                self.params = copy.deepcopy(params)
                path = tmp_path / (uuid.uuid4().hex+".wav")
                path.write_bytes(b"wave")
                return {"audioPath": str(path), "seed": params["seed"], "generationSettings": params["generationSettings"]}
            return {"cudaMemoryAllocatedBytes": 123}
    monkeypatch.setattr(module, "AuditionWorker", Worker)
    args = [vid, FP32, str(model), "cuda:0", "测试", "base", 2, 17, {"cfg": .5}, "all", 4, "native", uuid.uuid4().hex]
    wav, details, result = service.audition(*args)
    assert Path(wav).exists() and result["packSha256"] == sha256_file(source)
    assert instances[0].params["durationFactor"] == .5
    assert instances[0].params["generationSettings"] == {"cfg": .5}
    instances[0].failed = True
    with pytest.raises(RuntimeError, match="simulated"):
        service.audition(*args)
    assert instances[0].closed and Path(wav).exists()
    service.audition(*args)
    assert len(instances) == 2
    assert not Path(wav).exists()
    assert len(list((service.library.directory(vid)/(FP32+"-preview")).glob("*.wav"))) == 1
    service.close()


def test_gradio_builds_all_three_tabs_without_model_loading(tmp_path):
    pytest.importorskip("gradio")
    from indextts.validation_web import create_app
    app = create_app(source_model_dir="missing", emotion_model_dir="missing", output_dir=str(tmp_path/"out"), workspace_dir=str(tmp_path/"work"))
    labels = [c["props"].get("label") for c in app.config["components"] if c["type"] == "tabitem"]
    assert labels == ["音色包生成", "情感推理", "合成试听"]


def test_queued_cancel_does_not_start_worker(tmp_path, monkeypatch):
    import indextts.audition_service as module
    source = pack(tmp_path/"voice.ivp")
    service = WorkbenchService(tmp_path/"out", tmp_path/"work")
    service.library.install(source)
    vid = load_voicepack(source).voice_id
    model = tmp_path/"model"
    model.mkdir()
    (model/"runtime_model.json").write_text(json.dumps({"runtimeAbi":"indextts2.5-reader-runtime-v2", "sourceModelFingerprint":"a"*64}))
    monkeypatch.setattr(module, "AuditionWorker", lambda *a: pytest.fail("Cancelled queued task started a worker"))
    session = uuid.uuid4().hex
    errors = []
    def run():
        try:
            service.audition(vid, FP32, str(model), "cuda:0", "text", "base", 1, 17, {}, "all", 4, "native", session)
        except Exception as exc:
            errors.append(str(exc))
    audition = service.audition_service
    with service.gpu.use("audition"):
        thread = threading.Thread(target=run)
        thread.start()
        import time
        until = time.monotonic() + 3
        while not audition.has_task(session) and time.monotonic() < until:
            time.sleep(.01)
        assert audition.has_task(session)
        assert "请求取消" in service.cancel(session)
    thread.join(timeout=3)
    assert not thread.is_alive() and errors == ["试听已取消"]
    assert not audition.has_task(session)
    service.close()


def test_missing_cuda_is_explicit_and_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        ReaderRuntime(tmp_path/"missing-model", [], None, cache_dir=tmp_path/"cache")


def test_mismatched_model_and_bad_manual_vector_fail_without_loading(tmp_path, monkeypatch):
    import indextts.audition_service as module
    source = pack(tmp_path/"voice.ivp")
    service = WorkbenchService(tmp_path/"out", tmp_path/"work")
    service.library.install(source)
    vid = load_voicepack(source).voice_id
    model = tmp_path/"model"
    model.mkdir()
    (model/"runtime_model.json").write_text(json.dumps({"runtimeAbi":"indextts2.5-reader-runtime-v3", "sourceModelFingerprint":"a"*64}))
    monkeypatch.setattr(module, "AuditionWorker", lambda *a: pytest.fail("Invalid request loaded a model"))
    args = [vid, FP32, str(model), "cuda:0", "text", "base", 1, 17, {}, "all", 4, "native", uuid.uuid4().hex]
    with pytest.raises(ValueError, match="精度"):
        service.audition(*args)
    args[5] = [float("nan")]*8
    with pytest.raises(ValueError, match="情感向量"):
        service.audition(*args)
    assert not service.audition_service.has_task(args[-1])
    service.close()
