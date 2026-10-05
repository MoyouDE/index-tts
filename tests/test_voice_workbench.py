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


def test_dual_pack_profile_selection_metadata_and_library_import(tmp_path):
    from indextts.voicepack.archive import combine_voicepacks, extract_voicepack, repackage_voicepack
    import zipfile
    first=pack(tmp_path/'fp32.ivp',FP32,value=.25)
    second=pack(tmp_path/'bf16.ivp',BF16,value=.75)
    dual=combine_voicepacks(tmp_path/'dual.ivp',first,second)
    repeated=combine_voicepacks(tmp_path/'repeated.ivp',first,second)
    assert dual.read_bytes()==repeated.read_bytes()
    for profile,original,value,dtype in ((FP32,first,.25,torch.float32),(BF16,second,.75,torch.bfloat16)):
        selected=load_voicepack(dual,profile=profile,expected_model_fingerprint='a'*64)
        assert selected.available_profiles==(FP32,BF16) and selected.container_manifest['schemaVersion']==4
        assert selected.manifest['provenance']['profile']==profile
        assert selected.tensors['base_emotion'].dtype==dtype and selected.tensors['base_emotion'][0,0]==value
        output=extract_voicepack(dual,tmp_path/(profile+'.ivp'),profile)
        with zipfile.ZipFile(output) as a,zipfile.ZipFile(original) as b:
            assert a.read('conditioning.safetensors')==b.read('conditioning.safetensors')
            assert len(a.namelist())==5
        runtime=ReaderRuntime.__new__(ReaderRuntime)
        runtime.profile=profile;runtime.source_fingerprint='a'*64;runtime._voice_lock=threading.Lock()
        runtime.manifest={'voiceCompatibility':{k:selected.manifest['provenance'][k] for k in
            ('profile','referenceEncoderFingerprint','preprocessFingerprint')}}
        runtime.load_voice(dual)
        assert runtime.snapshot_voice(selected.voice_id).tensors['base_emotion'][0,0]==value
        runtime.voice_dirs=[tmp_path/'runtime-voices'];runtime.voice_dirs[0].mkdir(exist_ok=True)
        import shutil
        shutil.copyfile(dual,runtime.voice_dirs[0]/'voice.ivp')
        assert runtime.reload_voices()[0]['voiceId']==selected.voice_id
    renamed=repackage_voicepack(dual,tmp_path/'renamed.ivp','new name','male')
    with zipfile.ZipFile(dual) as a,zipfile.ZipFile(renamed) as b:
        for name in a.namelist():
            if not name.endswith('manifest.json'):assert a.read(name)==b.read(name)
    for profile in (FP32,BF16):
        selected=load_voicepack(renamed,profile=profile)
        assert selected.manifest['displayName']=='new name' and selected.manifest['gender']=='male'
    library=VoiceWorkspace(tmp_path/'dual-library'); library.install(renamed)
    assert all(v['ready'] for v in library.items()[0]['variants'].values())
    assert load_voicepack(library.pack_path(selected.voice_id,BF16)).manifest['provenance']['profile']==BF16


@pytest.mark.parametrize('bad', ['source','reference','name','profile','license','corrupt-inactive','unknown-profile'])
def test_dual_pack_rejects_inconsistent_or_corrupt_profiles(tmp_path,bad):
    from indextts.voicepack.archive import combine_voicepacks, VoicePackError
    import zipfile
    first=pack(tmp_path/'fp32.ivp',FP32)
    second=pack(tmp_path/'bf16.ivp',BF16)
    if bad in {'source','reference','name','profile','license'}:
        selected=load_voicepack(second); manifest=copy.deepcopy(selected.manifest);tensors=selected.tensors;licenses=_licenses()
        if bad=='source':manifest['sourceModelFingerprint']='d'*64
        if bad=='reference':manifest['provenance']['referenceSha256']='d'*64
        if bad=='name':manifest['displayName']='another voice'
        if bad=='profile':second=first
        else:
            if bad=='license':licenses['LICENSE']=b'changed license'
            second=write_voicepack(tmp_path/'changed.ivp',manifest,tensors,licenses)
        with pytest.raises(VoicePackError):combine_voicepacks(tmp_path/'dual.ivp',first,second)
        assert not (tmp_path/'dual.ivp').exists()
    else:
        dual=combine_voicepacks(tmp_path/'dual.ivp',first,second)
        if bad=='unknown-profile':
            with pytest.raises(VoicePackError,match='未知'):load_voicepack(dual,profile='fp16')
        else:
            with zipfile.ZipFile(dual) as archive:raw={n:archive.read(n) for n in archive.namelist()}
            raw['bf16/conditioning.safetensors']+=b'corrupt'
            with zipfile.ZipFile(dual,'w') as archive:
                for n,data in raw.items():archive.writestr(n,data)
            # Selecting FP32 must still validate the inactive BF16 payload.
            with pytest.raises(VoicePackError,match='哈希'):load_voicepack(dual,profile=FP32)


def test_precision_builds_share_identical_cropped_reference_snapshot(tmp_path,monkeypatch):
    import numpy as np
    import soundfile as sf
    from indextts.material_service import MaterialService, write_json
    from indextts.material_web import prepare_selection
    service=WorkbenchService(tmp_path/'jobs',tmp_path/'library',modules=('producer',))
    materials=MaterialService(tmp_path/'materials')
    source=uuid.uuid4().hex; directory=materials.directory(source); directory.mkdir()
    sf.write(directory/'audio.wav',np.zeros(22050*5),22050)
    record=dict(sourceId=source,sourceSha256='a'*64,audioSha256=sha256_file(directory/'audio.wav'),
        durationSeconds=5,primary='a',segments=[dict(id='a',start=1,end=4,selected=True)])
    record['revision']=materials.selection_revision(record);write_json(directory/'material.json',record)
    selection,_=prepare_selection(materials,source,[['a',1,4,True]],'a',record['revision'])
    hashes=[];cropped=[];crop=materials.crop
    def tracked_crop(*args):
        cropped.append(args);return crop(*args)
    monkeypatch.setattr(materials,'crop',tracked_crop)
    def fake_build(reference,vid,name,gender,profile,*args,**kwargs):
        hashes.append(sha256_file(reference))
        job=service.output_dir/uuid.uuid4().hex;job.mkdir(parents=True)
        return str(pack(job/'voice.ivp',profile,hashes[-1])),{}
    monkeypatch.setattr(service,'build',fake_build)
    try:
        first,_=service.build_selection(selection,'reader-female-01','阅读女声','female',FP32,'auto','unused',uuid.uuid4().hex,
                                      materials=materials,temporary=True)
        service.build_selection(selection,'reader-female-01','阅读女声','female',BF16,'auto','unused',uuid.uuid4().hex,
                                materials=materials,temporary=True,reference_snapshot=Path(first).parent/'references')
        assert len(cropped)==1 and len(hashes)==2 and hashes[0]==hashes[1]
    finally:materials.close();service.close()


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


def test_material_recipe_is_fixed_across_profiles_and_source_edits(tmp_path,monkeypatch):
    reference=tmp_path/"primary.wav"
    reference.write_bytes(b"fixed selected audio")
    digest=sha256_file(reference)
    selection={"method":"primary-only-v1","primary":"s002","primarySha256":digest,
               "revision":"old-selection","segments":[{"id":"s002","start":5.,"end":12.,"selected":True}]}
    source=pack(tmp_path/"first.ivp",reference_hash=digest)
    store=VoiceWorkspace(tmp_path/"work")
    installed=store.install(source,reference=reference,reference_selection=selection)
    vid=load_voicepack(source).voice_id
    record=store.record(vid)
    assert record["referenceSelection"]==selection
    owner=WorkbenchService(tmp_path/"out",store.root,modules="producer")
    observed=[]
    def build(ref,*args,reference_selection=None,**kwargs):
        observed.append((sha256_file(ref),copy.deepcopy(reference_selection)))
        return "new.ivp",{}
    monkeypatch.setattr(owner,"build",build)
    try:
        owner.rebuild(vid,BF16,"auto","unused",uuid.uuid4().hex)
        assert observed==[(digest,selection)]
        old_bytes=Path(installed).read_bytes()
        changed=copy.deepcopy(selection);changed["revision"]="edited-selection"
        with pytest.raises(ValueError,match="参考选择"):
            store.install(source,expected_instance=record["instance"],reference_selection=changed)
        assert Path(installed).read_bytes()==old_bytes
        low=pack(tmp_path/"low.ivp",BF16,digest)
        store.install(low,expected_instance=record["instance"],reference_selection=selection)
        assert all(v["ready"] for v in store.items()[0]["variants"].values())
    finally:owner.close()


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
