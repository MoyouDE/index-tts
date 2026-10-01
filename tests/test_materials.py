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
