import copy
import json
from pathlib import Path
import uuid
import numpy as np
import pytest
import soundfile as sf

from indextts.material_service import MaterialService, write_json
from indextts.material_vad import speech_intervals
from indextts.material_web import edit_split, edit_merge


@pytest.fixture
def material(tmp_path):
    service=MaterialService(tmp_path)
    sid=uuid.uuid4().hex
    directory=service.directory(sid)
    directory.mkdir()
    sf.write(directory/"audio.wav", np.linspace(-.2,.2,20*22050,dtype=np.float32),22050,subtype="FLOAT")
    from indextts.voicepack.provenance import sha256_file
    record={"sourceId":sid,"sourceSha256":"a"*64,"audioSha256":sha256_file(directory/"audio.wav"),
        "durationSeconds":20,"speechSeconds":15,"primary":"a","segments":[
            {"id":"a","start":1.,"end":7.,"selected":True},
            {"id":"b","start":9.,"end":18.,"selected":True}]}
    record["revision"]=service.selection_revision(record)
    write_json(directory/"material.json",record)
    return service,record


def rows(record):
    return [[s["id"],s["start"],s["end"],s["selected"]] for s in record["segments"]]


def test_vad_preserves_short_gaps_splits_long_gaps_and_bounds_length():
    values=[0]*10+[.9]*100+[0]*8+[.9]*100+[0]*25+[.9]*600+[0]*30
    clips, raw=speech_intervals(values,len(values)*.032)
    assert len(raw)==2
    assert raw[0][1]-raw[0][0] > 6
    assert len(clips)==3
    assert all(0<s["end"]-s["start"]<=15 for s in clips)
    assert all(a["end"]<=b["start"] for a,b in zip(clips,clips[1:]))
    assert speech_intervals([0]*100,3.2)[0]==[]
    with pytest.raises(ValueError): speech_intervals([float("nan")],.032)


def test_saved_selection_recovers_and_snapshot_is_detached(material):
    service,record=material
    saved=service.save(record["sourceId"],rows(record),"b",expected_revision=record["revision"])
    recovered=MaterialService(service.root.parent).record(record["sourceId"])
    assert recovered==saved
    snapshot=service.selection(saved["sourceId"],saved["revision"],confirmed=True)
    snapshot["segments"][0]["start"]=4
    assert service.record(saved["sourceId"])["segments"][0]["start"]==1
    with pytest.raises(ValueError,match="确认"): service.selection(saved["sourceId"],saved["revision"])
    with pytest.raises(ValueError,match="其他页面"): service.save(saved["sourceId"],rows(saved),"a",expected_revision=record["revision"])


def test_workflow_saves_selection_and_chooses_method_without_changing_primary_abi(material):
    from indextts.material_web import prepare_selection
    service, record = material
    selection, saved = prepare_selection(service, record["sourceId"], rows(record), "auto",
                                         record["revision"], True, "auto")
    assert selection["method"] == "speaker-mean-v1" and selection["primary"] == "b"
    assert selection["revision"] == saved["revision"]
    values = rows(saved)
    values[1][3] = False
    # Removing a manual primary recommends a remaining selected reference.
    selection, saved = prepare_selection(service, record["sourceId"], values, "b",
                                         saved["revision"], True, "auto")
    assert selection["method"] == "primary-only-v1" and selection["primary"] == "a"
    assert service.record(record["sourceId"]) == saved
    with pytest.raises(ValueError, match="其他页面"):
        prepare_selection(service, record["sourceId"], values, "auto", record["revision"], True, "auto")


def test_workflow_rejects_unconfirmed_overlong_or_invalid_fusion_without_saving(material):
    from indextts.material_web import prepare_selection
    service, record = material
    for values, approved, method, error in [
        (rows(record), False, "auto", "确认"),
        (edit_merge(rows(record), "a"), True, "auto", "拆分"),
        ([["a", 1., 7., True]], True, "speaker-mean-v1", "至少"),
    ]:
        with pytest.raises(ValueError, match=error):
            prepare_selection(service, record["sourceId"], values, "auto", record["revision"], approved, method)
        assert service.record(record["sourceId"]) == record


def test_workflow_encoding_failure_can_retry_and_queued_settings_are_fixed(material, tmp_path, monkeypatch):
    from indextts.validation_web import create_app
    service, record = material
    record["name"] = "reference.wav"
    write_json(service.directory(record["sourceId"])/"material.json", record)
    app = create_app(modules="producer", workspace_dir=service.root.parent, output_dir=tmp_path/"output")
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    captured = []
    def encode(selection, *args):
        captured.append((copy.deepcopy(selection), args))
        if len(captured) == 1:
            raise RuntimeError("model unavailable")
        return "voice.ivp", {"seconds": 1, "manifest": {"voiceId": args[0], "provenance": {"profile": args[3]}}}
    monkeypatch.setattr(app.workbench, "build_selection", encode)
    try:
        sid = uuid.uuid4().hex
        prepared, revision, _ = callbacks["prepare"](record["sourceId"], rows(record), "auto",
            record["revision"], True, "auto", json.dumps({"sourceId": record["sourceId"], "rows": rows(record), "primary": "auto"}),
            "Voice", "unknown", "compatible-fp32", "cpu", sid)
        with pytest.raises(Exception, match="model unavailable"):
            callbacks["generate_snapshot"](prepared)
        # A retry uses the revision already returned before encoding failed.
        retry, _, _ = callbacks["prepare"](record["sourceId"], rows(record), "auto",
            revision, True, "auto", json.dumps({"sourceId": record["sourceId"], "rows": rows(record), "primary": "auto"}),
            "Voice", "unknown", "compatible-fp32", "cpu", sid)
        result = callbacks["generate_snapshot"](retry)
        assert result[0] == "voice.ivp"
        selected, profile = callbacks["select_generated"](result[1])
        assert selected["value"] == captured[1][1][0] and profile == "compatible-fp32"
        assert captured[0][0] == captured[1][0] == prepared["selection"]
        assert captured[0][1][3:5] == ("compatible-fp32", "cpu")
        assert captured[0][1][0] != captured[1][1][0]  # distinct automatic voice IDs
    finally:
        app.workbench.close()


def test_manual_ranges_add_adjust_and_keep_excluded_history():
    from indextts.material_web import keep_range
    values = [["a", 1., 4., True], ["excluded", 5., 6., False]]
    updated, active = keep_range(values, "__new__", 7., 9., 20.)
    assert updated[-1] == [active, 7., 9., True]
    assert values == [["a", 1., 4., True], ["excluded", 5., 6., False]]
    adjusted, same = keep_range(updated, active, 5., 8., 20.)
    assert same == active and adjusted[-1] == [active, 5., 8., True]
    assert adjusted[1] == values[1]  # excluded ranges do not block an intentional replacement


@pytest.mark.parametrize("start,end,error", [(-1, 4, "范围"), (4, 4, "起点"),
    (1, 21, "范围"), (0, 16, "15"), (float("nan"), 4, "范围"), (3, 8, "重叠")])
def test_invalid_manual_range_does_not_change_draft(start, end, error):
    from indextts.material_web import keep_range
    values = [["a", 1., 4., True]]
    with pytest.raises(ValueError, match=error):
        keep_range(values, "__new__", start, end, 20.)
    assert values == [["a", 1., 4., True]]


def test_intelligent_suggestions_leave_saved_manual_selection_unchanged(material):
    service, record = material
    (service.directory(record["sourceId"])/"vad.wav").write_bytes(b"test")
    class Detector:
        def detect(self, *args):
            return {"segments": [{"id": "suggested", "start": 2., "end": 4., "selected": True}]}
    service._detector = Detector()
    suggestions = service.suggest_segments(record["sourceId"])
    assert suggestions["segments"][0]["id"] == "suggested"
    assert service.record(record["sourceId"]) == record


def test_browser_draft_is_authoritative_and_source_switch_is_rejected(material, tmp_path):
    from indextts.validation_web import create_app
    service, record = material
    record["name"] = "reference.wav"
    write_json(service.directory(record["sourceId"])/"material.json", record)
    app = create_app(modules="producer", workspace_dir=service.root.parent, output_dir=tmp_path/"out")
    prepare = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "prepare")
    try:
        with pytest.raises(Exception, match="不一致"):
            prepare(record["sourceId"], rows(record), "a", record["revision"], True, "auto",
                    json.dumps({"sourceId": "other", "rows": rows(record), "primary": "a"}),
                    "Voice", "unknown", "compatible-fp32", "cpu", uuid.uuid4().hex)
        assert service.record(record["sourceId"]) == record
        # A recent graphical edit may precede table mirroring; generation uses that edit.
        edited = [["a", 2., 7., True], ["b", 9., 18., False]]
        job, _, _ = prepare(record["sourceId"], rows(record), "b", record["revision"], True, "auto",
                    json.dumps({"sourceId": record["sourceId"], "rows": edited, "primary": "a"}),
                    "Voice", "unknown", "compatible-fp32", "cpu", uuid.uuid4().hex)
        assert job["selection"]["segments"] == [{"id": "a", "start": 2., "end": 7., "selected": True}]
        assert job["selection"]["method"] == "primary-only-v1"
    finally:
        app.workbench.close()


def test_waveform_cache_bounds_size_preserves_samples_and_recovers(material):
    service, record = material
    audio = service.directory(record["sourceId"])/"audio.wav"
    before = audio.read_bytes()
    peaks = service.waveform(record["sourceId"])
    assert peaks["duration"] == 20 and 0 < len(peaks["low"]) <= 12000
    assert len(peaks["low"]) == len(peaks["high"])
    assert min(peaks["low"]) == pytest.approx(-.2) and max(peaks["high"]) == pytest.approx(.2)
    assert service.waveform(record["sourceId"]) == peaks
    (audio.parent/"waveform-v1.json").write_text("broken", encoding="utf-8")
    assert service.waveform(record["sourceId"]) == peaks
    assert audio.read_bytes() == before and service.record(record["sourceId"]) == record


@pytest.mark.parametrize("draft", ["", "null", "[]", '{"sourceId":"other"}',
    '{"sourceId":"test","rows":[["a",true,3,true]],"primary":"a"}'])
def test_invalid_graphical_draft_is_rejected(draft):
    from indextts.material_web import read_draft
    with pytest.raises(ValueError, match="选择未就绪"):
        read_draft("test", draft)


@pytest.mark.parametrize("bad", [
    [["a",-1,3,True]], [["a",0,21,True]], [["a",0,float("nan"),True]],
    [["a",0,6,True],["b",5,10,True]], [["a",0,6,True],["a",7,10,True]],
    [["a",0,6,"True"]], [["a",True,6,True]], [["a",0,6,False]],
])
def test_invalid_edits_preserve_record(material,bad):
    service,record=material
    with pytest.raises((ValueError,TypeError)):
        service.save(record["sourceId"],bad,"a",expected_revision=record["revision"])
    assert service.record(record["sourceId"])==record


def test_overlong_selection_requires_split_and_crop_never_truncates(material,tmp_path):
    service,record=material
    merged=edit_merge(rows(record),"a")
    saved=service.save(record["sourceId"],merged,"a",expected_revision=record["revision"])
    with pytest.raises(ValueError,match="拆分"):
        service.selection(saved["sourceId"],saved["revision"],confirmed=True)
    split=edit_split(merged,"a",9)
    saved=service.save(record["sourceId"],split,"a",expected_revision=saved["revision"])
    assert len(service.selection(saved["sourceId"],saved["revision"],confirmed=True)["segments"])==2
    with pytest.raises(ValueError): service.crop(saved["sourceId"],0,20,tmp_path/"bad.wav")
    cropped=service.crop(saved["sourceId"],1,7,tmp_path/"clip.wav")
    assert sf.info(cropped).duration==6
    with sf.SoundFile(service.directory(saved["sourceId"])/"audio.wav") as original:
        original.seek(22050)
        assert np.array_equal(original.read(6*22050,dtype="float32"),sf.read(cropped,dtype="float32")[0])


def test_identity_fusion_only_changes_identity_pair():
    import torch
    from indextts.voicepack.experimental_fusion import fuse_identity
    pack={"speaker_style":torch.tensor([[1.,2.]]),"speaker_latent":torch.tensor([[7.]]),
          "base_emotion":torch.tensor([[3.]]),"emotion_basis":torch.ones(8,1),
          "prompt_condition":torch.ones(1,3,2),"ref_mel":torch.ones(1,2,3)}
    result=fuse_identity(pack,[pack["speaker_style"],torch.tensor([[3.,2.]])],torch.tensor([[2.,1.]]),torch.tensor([1.]),"mean")
    assert torch.equal(result["speaker_style"],torch.tensor([[2.,2.]]))
    assert torch.equal(result["speaker_latent"],torch.tensor([[7.]]))
    for name in set(pack)-{"speaker_style","speaker_latent"}:
        assert torch.equal(pack[name],result[name])
    with pytest.raises(ValueError): fuse_identity(pack,[torch.zeros(1,2)],torch.ones(1,2),torch.zeros(1),"normalized")


def test_normalized_mean_restores_mean_input_norm_without_normalizing_output():
    import torch
    from indextts.voicepack.experimental_fusion import fuse_identity
    primary={"speaker_style":torch.tensor([[3.,0.]]),"speaker_latent":torch.zeros(1,2)}
    result=fuse_identity(primary,[primary["speaker_style"],torch.tensor([[0.,5.]])],
                         torch.eye(2),torch.zeros(2),"normalized")
    assert torch.equal(result["speaker_style"],torch.tensor([[2.,2.]]))
    assert torch.equal(result["speaker_latent"],result["speaker_style"])


def test_decode_failure_does_not_install_partial_material_and_next_import_recovers(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import indextts.material_service as module
    class Detector:
        def detect(self,*args):
            return {"segments":[{"id":"a","start":0.,"end":1.,"selected":True}],
                    "speechSeconds":1.,"speechIntervals":[[0.,1.]]}
    service=MaterialService(tmp_path/"work",detector=Detector())
    source=tmp_path/"source.mp4";source.write_bytes(b"fixture")
    monkeypatch.setattr(module,"ffmpeg_executable",lambda:"ffmpeg")
    monkeypatch.setattr(module.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stderr="decode error"))
    with pytest.raises(ValueError,match="decode error"):service.import_media(source)
    assert not list(service.root.iterdir())
    def decode(command,**kwargs):
        sf.write(command[-1],np.zeros(22050,dtype=np.float32),22050,subtype="FLOAT")
        return SimpleNamespace(returncode=0,stderr="")
    monkeypatch.setattr(module.subprocess,"run",decode)
    imported=service.import_media(source)
    assert service.record(imported["sourceId"])==imported
    assert (service.directory(imported["sourceId"])/"original.mp4").read_bytes()==b"fixture"


def test_selection_build_fixes_primary_audio_before_waiting_for_producer(material,tmp_path,monkeypatch):
    from indextts.workbench_service import WorkbenchService
    from indextts.voicepack.provenance import sha256_file
    service,record=material
    owner=WorkbenchService(tmp_path/"out",service.root.parent,modules="producer")
    selection=service.selection(record["sourceId"],record["revision"],confirmed=True)
    observed=[]
    def build(reference,*args,reference_selection=None,**kwargs):
        # An edit after submission cannot change this already-cropped reference.
        service.save(record["sourceId"],rows(record),"b",expected_revision=record["revision"])
        observed.append((sf.info(reference).duration,sha256_file(reference),copy.deepcopy(reference_selection)))
        return "voice.ivp",{}
    monkeypatch.setattr(owner,"build",build)
    try:
        owner.build_selection(selection,"voice","Voice","unknown","compatible-fp32","cpu","missing",uuid.uuid4().hex)
        assert observed[0][0]==6
        assert observed[0][1]==observed[0][2]["primarySha256"]
        assert observed[0][2]["primary"]=="a" and selection["primary"]=="a"
        with pytest.raises(ValueError,match="变更"):
            owner.build_selection(selection,"other","Voice","unknown","compatible-fp32","cpu","missing",uuid.uuid4().hex)
    finally:owner.close()


def test_blind_comparisons_keep_identity_and_pause_questions_separate(tmp_path):
    from indextts.material_service import digest_json
    from tools.reference_fusion_experiment import make_blind
    config={"recipe":{},"recipeSha256":digest_json({}),"clipSha256":{},"texts":["hello"],"seeds":[17],
            "cases":["baseline","mean-3","pause-keep","pause-short"]}
    write_json(tmp_path/"configuration.json",config)
    for case in config["cases"]:
        (tmp_path/case).mkdir()
    def sample(case,amplitude):
        sf.write(tmp_path/case/"seed-17-text-1.wav",np.full(1600,amplitude,dtype=np.float32),16000,subtype="FLOAT")
    sample("baseline",.2);sample("mean-3",.4)
    before=(tmp_path/"mean-3/seed-17-text-1.wav").read_bytes()
    assert make_blind(tmp_path)==["identity"]
    assert not (tmp_path/"blind/pause-seed-17-text-1.wav").exists()
    sample("pause-keep",.1);sample("pause-short",.3)
    assert make_blind(tmp_path)==["identity","pause"]
    key=json.loads((tmp_path/"blind-key.json").read_text(encoding="utf-8"))
    assert {c for label,c in key.items() if label.startswith("V")}=={"baseline","mean-3"}
    assert {c for label,c in key.items() if label.startswith("P")}=={"pause-keep","pause-short"}
    assert sf.info(tmp_path/"blind/identity-seed-17-text-1.wav").duration==2.2
    for label,case in key.items():
        result,rate=sf.read(tmp_path/"blind"/(label+"-seed-17-text-1.wav"))
        assert rate==16000 and len(result)==1600
        assert np.max(result)<=.20001
    assert (tmp_path/"mean-3/seed-17-text-1.wav").read_bytes()==before
