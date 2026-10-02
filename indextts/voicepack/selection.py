"""Versioned reference selection and identity-only conditioning construction."""
import math
import re
from pathlib import Path
from .schema import VoicePackSchemaError
from .provenance import sha256_file

METHODS = {"primary-only-v1", "speaker-mean-v1"}
HASH = re.compile(r"^[0-9a-f]{64}$")

def validate_selection_record(selection, primary_hash):
    keys = {"sourceId", "sourceSha256", "audioSha256", "revision", "segments", "primary", "method", "confirmedTarget", "primarySha256"}
    if not isinstance(selection, dict) or set(selection) != keys or selection["method"] not in METHODS:
        raise VoicePackSchemaError("未知参考条件构建方法或来源字段")
    if selection["confirmedTarget"] is not True:
        raise VoicePackSchemaError("参考人物未确认")
    for key in ("sourceSha256", "audioSha256", "revision", "primarySha256"):
        if not isinstance(selection[key], str) or not HASH.fullmatch(selection[key]):
            raise VoicePackSchemaError("参考来源哈希无效")
    if selection["primarySha256"] != primary_hash:
        raise VoicePackSchemaError("主参考哈希不一致")
    segments = selection["segments"]
    if not isinstance(segments, list) or not segments or len(segments)>128:
        raise VoicePackSchemaError("参考片段数量无效")
    ids=set(); previous=-1
    for seg in segments:
        if not isinstance(seg, dict) or set(seg)!={"id", "start", "end", "selected", "sha256"}:
            raise VoicePackSchemaError("参考片段字段无效")
        if not isinstance(seg["id"],str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",seg["id"]) or seg["id"] in ids:
            raise VoicePackSchemaError("参考片段 ID 无效或重复")
        ids.add(seg["id"])
        a,b=seg["start"],seg["end"]
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (a,b)) or not 0<=a<b or b-a>15+1e-6 or a<previous-1e-6:
            raise VoicePackSchemaError("参考片段范围无效、重叠或超过 15 秒")
        previous=b
        if seg["selected"] is not True or not isinstance(seg["sha256"],str) or not HASH.fullmatch(seg["sha256"]):
            raise VoicePackSchemaError("参考片段哈希或选择无效")
    if selection["primary"] not in ids or next(s["sha256"] for s in segments if s["id"]==selection["primary"]) != primary_hash:
        raise VoicePackSchemaError("主参考不在片段中或哈希不一致")
    if selection["method"]=="speaker-mean-v1" and len(segments)<2:
        raise VoicePackSchemaError("身份融合至少需要两个不同片段")

def validate_selection(selection, references, primary):
    validate_selection_record(selection,sha256_file(primary))
    if not isinstance(references,dict) or set(references)!={s["id"] for s in selection["segments"]}:
        raise ValueError("参考快照片段集合不一致")
    import soundfile as sf
    for seg in selection["segments"]:
        path=Path(references[seg["id"]]);info=sf.info(path)
        if info.duration>15+1e-6 or abs(info.duration-(seg["end"]-seg["start"]))>2/info.samplerate:
            raise ValueError("参考快照时长不一致")
        if sha256_file(path)!=seg["sha256"]:
            raise ValueError("参考快照哈希不一致")

def fuse_selected(encoder, primary, references, primary_id):
    import torch
    from .reference import staged_model
    styles=[]
    for sid,path in references.items():
        if sid==primary_id: styles.append(primary["speaker_style"])
        else: styles.append(encoder.extract_speaker_style(str(path)))
    style=torch.stack(styles).mean(0)
    with torch.no_grad(), staged_model(encoder.projection,encoder.device) as projection:
        with torch.amp.autocast(encoder.device.type,enabled=encoder.dtype is not None,dtype=encoder.dtype):
            latent=projection.spk_emb_proj(style.to(encoder.device))
    return {**primary,"speaker_style":style.cpu().contiguous(),"speaker_latent":latent.cpu().contiguous()}
