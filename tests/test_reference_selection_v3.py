import copy
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from indextts.voicepack.archive import load_voicepack,write_voicepack,VoicePackError
from indextts.voicepack.provenance import sha256_file
from indextts.voicepack.selection import validate_selection,fuse_selected
from indextts.voice_workspace import VoiceWorkspace

@pytest.fixture
def selected(tmp_path):
    root=Path(__file__).parent/"fixtures/voices-fp32"
    old=load_voicepack(next(root.glob("*.ivp")))
    manifest=copy.deepcopy(old.manifest)
    from indextts.voicepack.provenance import PREPROCESS_FINGERPRINT
    refs={}
    segments=[]
    for i in range(2):
        sid="r"+str(i);path=tmp_path/(sid+".wav")
        sf.write(path,np.full(22050,.1*(i+1),dtype=np.float32),22050,subtype="FLOAT")
        refs[sid]=str(path)
        segments.append(dict(id=sid,start=i,end=i+1,selected=True,sha256=sha256_file(path)))
    selection=dict(sourceId="a"*32,sourceSha256="b"*64,audioSha256="c"*64,revision="d"*64,segments=segments,primary="r0",method="speaker-mean-v1",confirmedTarget=True,primarySha256=segments[0]["sha256"])
    manifest["schemaVersion"]=3
    manifest["provenance"]=dict(profile="compatible-fp32",referenceSha256=selection["primarySha256"],referenceEncoderFingerprint="e"*64,preprocessFingerprint=PREPROCESS_FINGERPRINT,producerVersions={},referenceSelection=selection)
    licenses={n:b"test" for n in ("LICENSE","LICENSE_ZH.txt","DERIVATIVE_DISCLAIMER.txt")}
    return manifest,old.tensors,licenses,selection,refs

def test_v3_roundtrip_and_library_recovery(selected,tmp_path):
    manifest,tensors,licenses,selection,refs=selected
    validate_selection(selection,refs,refs["r0"])
    pack=write_voicepack(tmp_path/"v3.ivp",manifest,tensors,licenses)
    assert load_voicepack(pack).manifest["provenance"]["referenceSelection"]==selection
    library=VoiceWorkspace(tmp_path/"library")
    library.install(pack,reference=refs["r0"],reference_selection=selection,references=refs)
    restored=VoiceWorkspace(library.root)
    record=restored.record(manifest["voiceId"])
    for seg in selection["segments"]:
        assert sha256_file(restored.directory(manifest["voiceId"])/"references"/(seg["id"]+".wav"))==seg["sha256"]
    bad=copy.deepcopy(selection);bad["method"]="primary-only-v1"
    with pytest.raises(ValueError):
        restored.install(pack,reference=refs["r0"],reference_selection=bad,expected_instance=record["instance"])

@pytest.mark.parametrize("fault",["method","hash","primary","overlap","overlong","duplicate"])
def test_rejects_bad_v3_provenance(selected,tmp_path,fault):
    manifest,tensors,licenses,selection,_=selected
    if fault=="method":selection["method"]="unknown"
    elif fault=="hash":selection["segments"][0]["sha256"]="0"*64
    elif fault=="primary":selection["primary"]="other"
    elif fault=="overlap":selection["segments"][1]["start"]=.5
    elif fault=="overlong":selection["segments"][1]["end"]=20
    else:selection["segments"][1]["id"]="r0"
    with pytest.raises(VoicePackError):write_voicepack(tmp_path/"bad.ivp",manifest,tensors,licenses)

def test_snapshot_tampering_rejected(selected):
    _,_,_,selection,refs=selected
    Path(refs["r1"]).write_bytes(b"broken")
    with pytest.raises((ValueError,RuntimeError)):validate_selection(selection,refs,refs["r0"])

def test_identity_only_fusion_uses_projection_and_keeps_other_tensors():
    from types import SimpleNamespace
    primary={"speaker_style":torch.ones(1,192),"prompt_condition":torch.randn(1,2,512),"ref_mel":torch.randn(1,80,2),"base_emotion":torch.randn(1,1280),"emotion_basis":torch.randn(8,1280)}
    projection=torch.nn.Module();projection.spk_emb_proj=torch.nn.Linear(192,1280)
    encoder=SimpleNamespace(projection=projection,device=torch.device("cpu"),dtype=None,extract_speaker_style=lambda *a,**k:torch.full((1,192),3.))
    result=fuse_selected(encoder,primary,{"r0":"one","r1":"two"},"r0")
    assert torch.equal(result["speaker_style"],torch.full((1,192),2.))
    assert torch.equal(result["speaker_latent"],projection.spk_emb_proj(result["speaker_style"]))
    for name in ("prompt_condition","ref_mel","base_emotion","emotion_basis"):assert torch.equal(result[name],primary[name])


@pytest.mark.parametrize("profile,dtype",[("compatible-fp32",torch.float32),("fixed-voice-bf16",torch.bfloat16)])
def test_v3_precision_roundtrip_and_mismatch_rejection(selected,tmp_path,profile,dtype):
    manifest,tensors,licenses,_,_=selected
    tensors={k:v.clone() for k,v in tensors.items()}
    manifest["provenance"]["profile"]=profile
    for k in ("speaker_latent","base_emotion"):tensors[k]=tensors[k].to(dtype)
    pack=write_voicepack(tmp_path/"precision.ivp",manifest,tensors,licenses)
    assert load_voicepack(pack).tensors["speaker_latent"].dtype==dtype
    manifest["provenance"]["profile"]="fixed-voice-bf16" if profile=="compatible-fp32" else "compatible-fp32"
    with pytest.raises(VoicePackError,match="precision mismatch"):write_voicepack(tmp_path/"wrong.ivp",manifest,tensors,licenses)


def test_rebuild_snapshots_all_fragments_before_original_is_deleted(selected,tmp_path,monkeypatch):
    import uuid
    from indextts.workbench_service import WorkbenchService
    manifest,tensors,licenses,selection,refs=selected
    pack=write_voicepack(tmp_path/"original.ivp",manifest,tensors,licenses)
    owner=WorkbenchService(tmp_path/"sessions",tmp_path/"workspace",modules="producer")
    owner.library.install(pack,reference=refs["r0"],reference_selection=selection,references=refs)
    vid=manifest["voiceId"];record=owner.library.record(vid)
    def consume(reference,*args,**kwargs):
        owner.library.delete(vid,vid+":"+record["instance"])
        validate_selection(kwargs["reference_selection"],kwargs["references"],reference)
        return "snapshot-still-intact",{}
    monkeypatch.setattr(owner,"build",consume)
    try:
        assert owner.rebuild(vid,"compatible-fp32","cpu","unused",uuid.uuid4().hex)[0]=="snapshot-still-intact"
    finally:owner.close()


def test_invalid_snapshot_fails_before_loading_models(selected,tmp_path,monkeypatch):
    from indextts.voicepack.builder import VoicePackBuilder
    _,_,_,selection,refs=selected
    builder=VoicePackBuilder(model_dir=tmp_path,profile="compatible-fp32",source_model_fingerprint="a"*64)
    def forbidden():raise AssertionError("invalid snapshot loaded models")
    monkeypatch.setattr(builder,"_get_tts",forbidden)
    selection["method"]="unknown"
    with pytest.raises(ValueError,match="构建方法"):
        builder.build(refs["r0"],{"voiceId":"voice","displayName":"Voice","gender":"unknown"},tmp_path/"bad.ivp",reference_selection=selection,references=refs)
