"""Reproducible, isolated FP32 reference experiments; never writes an IVP/library entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TEXTS=["今天的天气很好，我们一起去公园散步，顺便聊聊最近发生的事情。",
       "这段声音用于音色克隆对比。请留意发音是否清楚，音色是否稳定。",
       "会议安排在十月十二日下午三点半，请准备两份材料。到达之后，先核对姓名，再按照顺序入场。",
       "他停下脚步，想了想，然后轻声说：我们明天再讨论吧。窗外的雨还没有停，但房间里已经安静下来。"]
GENERATION={"do_sample":False,"num_beams":3,"repetition_penalty":10.,"length_penalty":0.,
            "max_generate_length":1500,"acoustic_steps":25,"cfg":.7}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path,value):
    from indextts.material_service import write_json
    write_json(Path(path),value)


def checked_configuration(output):
    from indextts.material_service import digest_json
    from indextts.voicepack.provenance import sha256_file
    config=read(output/"configuration.json")
    if digest_json(config["recipe"])!=config["recipeSha256"]:
        raise ValueError("实验选择记录哈希不匹配")
    for name,expected in config["clipSha256"].items():
        path=(output/"clips"/name).resolve()
        if path.parent!=(output/"clips").resolve() or sha256_file(path)!=expected:
            raise ValueError("实验参考片段完整性检查失败")
    return config


def initialize():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(17)
    return torch


def prepare(args):
    from indextts.material_service import MaterialService,digest_json
    from indextts.voicepack.provenance import sha256_file
    service=MaterialService(args.workspace)
    record=service.record(args.material_id)
    if not args.confirmed_target:
        raise ValueError("必须先人工试听并确认训练和留出片段属于同一目标人物，再传 --confirmed-target")
    ids=args.segments.split(",")
    held=args.held_out.split(",")
    if len(ids)!=len(set(ids)) or len(held)!=len(set(held)) or set(ids)&set(held) or not held or not ids:
        raise ValueError("训练与留出片段必须非空、各自不重复且不相交")
    if args.primary not in ids:
        raise ValueError("主参考必须在训练片段内")
    lookup={s["id"]:s for s in record["segments"]}
    selected=[lookup[i] for i in ids]
    held_out=[lookup[i] for i in held]
    ordered=sorted(selected+held_out,key=lambda s:s["start"])
    if any(s["end"]-s["start"]>15+1e-6 for s in ordered) or any(a["end"]>b["start"]+1e-6 for a,b in zip(ordered,ordered[1:])):
        raise ValueError("实验片段必须不重叠且每段不超过 15 秒")
    if not 0<=args.pause_start<args.pause_end<=record["durationSeconds"] or args.pause_end-args.pause_start>15:
        raise ValueError("空白对照区间须在素材内且不超过 15 秒")
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    clips=output/"clips";clips.mkdir()
    for segment in selected+held_out:
        service.crop(record["sourceId"],segment["start"],segment["end"],clips/(segment["id"]+".wav"))
    # Keep speech samples unchanged; only shorten internal long VAD gaps to .2s.
    import soundfile as sf
    import numpy as np
    keep=clips/"pause-keep.wav"
    service.crop(record["sourceId"],args.pause_start,args.pause_end,keep)
    wave,rate=sf.read(keep,dtype="float32")
    gaps=[]
    raw=record["speechIntervals"]
    for previous,following in zip(raw,raw[1:]):
        a,b=max(previous[1],args.pause_start),min(following[0],args.pause_end)
        if args.pause_start< a < b < args.pause_end and b-a>=.5:
            gaps.append((round((a+.1-args.pause_start)*rate),round((b-.1-args.pause_start)*rate)))
    parts=[];start=0
    for a,b in gaps:
        parts.append(wave[start:a]);start=b
    parts.append(wave[start:])
    sf.write(clips/"pause-short.wav",np.concatenate(parts),rate,subtype="FLOAT")
    ordered_ids=[args.primary]+[i for i in ids if i!=args.primary]
    counts=sorted({min(3,len(ids)),min(5,len(ids))}-{1})
    cases=["baseline"]+[f"{method}-{n}" for method in ("mean","normalized") for n in counts]
    if gaps: cases += ["pause-keep","pause-short"]
    recipe={"sourceId":record["sourceId"],"sourceSha256":record["sourceSha256"],"audioSha256":record["audioSha256"],
        "materialRevision":record["revision"],"segments":selected,"heldOut":held_out,"primary":args.primary,
        "pauseRange":[args.pause_start,args.pause_end],"removedGaps":gaps,"confirmedTarget":True,
        "pauseTargetConfirmed":args.confirmed_pause_target}
    config={"schemaVersion":1,"recipe":recipe,"recipeSha256":digest_json(recipe),"trainingIds":ordered_ids,
        "heldOutIds":held,"clipSha256":{p.name:sha256_file(p) for p in sorted(clips.glob("*.wav"))},
        "sourceModels":str(Path(args.source_model_dir).resolve()),"readerModels":str(Path(args.reader_model_dir).resolve()),
        "texts":TEXTS,"seeds":[17,29],"generationSettings":GENERATION,"cases":cases,
        "precision":"FP32","encoderDevice":"cpu","synthesisDevice":"cuda:0","cpuThreads":4,
        "scope":"One source recording; auxiliary CAMPPlus metrics are not subjective quality validation.",
        "pauseControlStatus":"applicable" if gaps else "no-internal-gap-at-least-500ms"}
    save(output/"configuration.json",config)
    print(json.dumps({"output":str(output),"cases":cases,"pauseControl":config["pauseControlStatus"]},ensure_ascii=False))


def style_for_audio(path,camp,torch):
    import torchaudio
    from indextts.voicepack.reference import load_reference_audio
    audio,rate=load_reference_audio(str(path))
    audio16=torchaudio.functional.resample(audio,rate,16000) if rate!=16000 else audio
    feat=torchaudio.compliance.kaldi.fbank(audio16,num_mel_bins=80,dither=0,sample_frequency=16000)
    feat=feat-feat.mean(dim=0,keepdim=True)
    with torch.no_grad(): return camp(feat.unsqueeze(0)).detach().cpu().contiguous()


def encode(output):
    torch=initialize()
    from safetensors.torch import save_file
    from indextts.voicepack.reference import ReferenceEncoder
    from indextts.voicepack.builder import model_fingerprint
    from indextts.voicepack.provenance import reference_fingerprint
    config=checked_configuration(output)
    encoder=ReferenceEncoder(model_dir=config["sourceModels"],device="cpu")
    started=time.perf_counter()
    try:
        primary=encoder.extract_voice_conditioning(str(output/"clips"/(config["recipe"]["primary"]+".wav")))
        save_file(primary,str(output/"baseline.safetensors"))
        # Bind the estimator and projection to the same strictly loaded checkpoint.
        styles={}
        for sid in config["trainingIds"]+config["heldOutIds"]:
            styles[sid]=style_for_audio(output/"clips"/(sid+".wav"),encoder.camp,torch)
        assert torch.equal(styles[config["recipe"]["primary"]],primary["speaker_style"])
        save_file(styles,str(output/"styles.safetensors"))
        from indextts.voicepack.experimental_fusion import fuse_identity
        weight,bias=encoder.projection.spk_emb_proj.weight.detach(),encoder.projection.spk_emb_proj.bias.detach()
        for case in config["cases"]:
            if case=="baseline": continue
            if case.startswith("pause-"):
                tensors=encoder.extract_voice_conditioning(str(output/"clips"/(case+".wav")))
            else:
                method,count=case.split("-")
                tensors=fuse_identity(primary,[styles[sid] for sid in config["trainingIds"][:int(count)]],weight,bias,method)
                for name in ("prompt_condition","ref_mel","base_emotion","emotion_basis"):
                    assert torch.equal(tensors[name],primary[name])
            if not all(torch.isfinite(t).all() and t.dtype==torch.float32 for t in tensors.values()):
                raise ValueError("非有限或非 FP32 条件张量")
            save_file(tensors,str(output/(case+".safetensors")))
        fingerprint=model_fingerprint(config["sourceModels"])
        assert fingerprint==read(Path(config["readerModels"])/"runtime_model.json")["sourceModelFingerprint"]
        save(output/"encoding.json",{"ok":True,"seconds":time.perf_counter()-started,
             "sourceModelFingerprint":fingerprint,"referenceEncoderFingerprint":reference_fingerprint(Path(config["sourceModels"]),encoder.cfg),
             "fixedAcousticPrompt":True,"artifactFormat":"experimental raw tensors; not IVP"})
    finally: encoder.close()


def silence_metrics(wave,rate):
    import numpy as np
    frame=max(1,round(rate*.02))
    n=len(wave)//frame
    if not n: raise ValueError("输出音频过短")
    rms=np.sqrt(np.mean(wave[:n*frame].reshape(n,frame)**2,axis=1))
    threshold=max(.001,float(rms.max())*.03)
    active=np.flatnonzero(rms>=threshold)
    if not len(active): raise ValueError("输出只有静音")
    gaps=[];start=None
    for i in range(int(active[0]),int(active[-1])+1):
        if rms[i]<threshold:
            if start is None: start=i
        elif start is not None:
            if (i-start)*.02>=.1: gaps.append((i-start)*.02)
            start=None
    return {"internalSilenceSeconds":sum(gaps),"longestInternalSilenceSeconds":max(gaps,default=0),
            "rmsThreshold":threshold,"silenceMetric":"20ms RMS frames; max(0.001, 3% peak RMS); internal gaps >=100ms",
            "clippingFraction":float(np.mean(np.abs(wave)>=32766/32768))}


def synthesize(output,case):
    torch=initialize()
    import soundfile as sf
    import psutil
    from safetensors.torch import load_file
    from indextts.runtime.engine import ReaderRuntime
    from indextts.voicepack.archive import VoicePack
    from indextts.runtime.profiles import InferenceOptimizations
    config=checked_configuration(output)
    if case not in config["cases"]: raise ValueError("未知实验组")
    if case.startswith("pause-") and config["recipe"].get("pauseTargetConfirmed") is not True:
        raise ValueError("空白对照区间尚未人工确认同一目标人物，不能生成该组")
    destination=output/case;destination.mkdir(exist_ok=True)
    raw=load_file(str(output/(case+".safetensors")))
    started=time.perf_counter()
    runtime=ReaderRuntime(config["readerModels"],[],None,cache_dir=destination/"cache",optimizations=InferenceOptimizations())
    if runtime.profile != "compatible-fp32": raise ValueError("首轮实验必须使用 FP32 裁剪模型")
    if runtime.source_fingerprint!=read(output/"encoding.json")["sourceModelFingerprint"]: raise ValueError("模型指纹不同")
    pack=VoicePack(output/(case+".safetensors"),{"voiceId":"experiment-"+case},raw)
    report={"case":case,"modelLoadSeconds":time.perf_counter()-started,"samples":[],"gpu":torch.cuda.get_device_name(0)}
    try:
        for seed in config["seeds"]:
            for i,text in enumerate(config["texts"],1):
                torch.cuda.reset_peak_memory_stats()
                result=runtime.synthesize(text,pack.voice_id,emotion="base",seed=seed,_voice_pack=pack,
                                          generation_settings=config["generationSettings"])
                audio=destination/f"seed-{seed}-text-{i}.wav"
                os.replace(result["audioPath"],audio)
                wave,rate=sf.read(audio,dtype="float32")
                if not len(wave) or not __import__('numpy').isfinite(wave).all(): raise ValueError("无效音频")
                item={"textIndex":i,"seed":seed,"file":audio.name,"result":{**result,"audioPath":audio.name},
                      "torchPeakAllocatedMiB":torch.cuda.max_memory_allocated()/1024**2,
                      "torchPeakReservedMiB":torch.cuda.max_memory_reserved()/1024**2,
                      "processRssMiB":psutil.Process().memory_info().rss/1024**2,
                      **silence_metrics(wave,rate)}
                report["samples"].append(item)
                save(destination/"synthesis.json",report)
                print(f'{case} seed {seed} text {i}: RTF {result["rtf"]:.3f}',flush=True)
    finally:
        del runtime


def make_blind(output):
    import numpy as np
    import soundfile as sf
    config=checked_configuration(output)
    blind=output/"blind";blind.mkdir(exist_ok=True)
    mapping={};available=[]
    groups={"identity":("V",[c for c in config["cases"] if not c.startswith("pause-")]),
            "pause":("P",[c for c in config["cases"] if c.startswith("pause-")])}
    for group,(prefix,cases) in groups.items():
        if not cases or any(not (output/c/f"seed-{seed}-text-{i}.wav").is_file()
                            for c in cases for seed in config["seeds"] for i in range(1,len(config["texts"])+1)):
            continue
        order=cases.copy();random.Random(917).shuffle(order)
        labels={f"{prefix}{i+1:02d}":case for i,case in enumerate(order)}
        mapping.update(labels);available.append(group)
        for seed in config["seeds"]:
            for text_index in range(1,len(config["texts"])+1):
                pieces=[];audio_info=[];loaded=[]
                for label,case in labels.items():
                    wave,rate=sf.read(output/case/f"seed-{seed}-text-{text_index}.wav",dtype="float32")
                    if not len(wave) or not np.isfinite(wave).all(): raise ValueError("无效试听音频")
                    loaded.append((label,wave,rate))
                if len({rate for _,_,rate in loaded}) != 1: raise ValueError("试听采样率不一致")
                target=min(float(np.sqrt(np.mean(w**2))) for _,w,_ in loaded)
                offset=0.
                for label,wave,rate in loaded:
                    gain=min(1.,target/max(1e-8,float(np.sqrt(np.mean(wave**2)))))
                    sf.write(blind/f"{label}-seed-{seed}-text-{text_index}.wav",wave*gain,rate)
                    audio_info.append({"label":label,"startSeconds":offset,"durationSeconds":len(wave)/rate})
                    pieces.extend([wave*gain,np.zeros(rate,dtype=np.float32)]);offset+=len(wave)/rate+1
                stem=f"{group}-seed-{seed}-text-{text_index}"
                sf.write(blind/(stem+".wav"),np.concatenate(pieces),rate)
                save(blind/(stem+".json"),audio_info)
    save(output/"blind-key.json",mapping)
    save(output/"blind-info.json",{"availableGroups":available,"identityLabels":"V01 onward",
         "pauseLabels":"P01 onward","subjectiveVerdict":"pending-user-listening"})
    return available


def summarize(output):
    torch=initialize()
    import numpy as np
    from safetensors.torch import load_file
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from torch.nn import functional as F
    config=checked_configuration(output)
    camp=CAMPPlus(feat_dim=80,embedding_size=192)
    camp.load_state_dict(torch.load(Path(config["sourceModels"])/"hf_cache/campplus_cn_common.bin",map_location="cpu"),strict=True)
    camp.eval()
    styles=load_file(str(output/"styles.safetensors"))
    held=[styles[sid] for sid in config["heldOutIds"]]
    results=[]
    for case in config["cases"]:
        path=output/case/"synthesis.json"
        if not path.exists(): raise ValueError("实验组尚未完成，不能生成完整技术报告")
        report=read(path)
        expected={(seed,i) for seed in config["seeds"] for i in range(1,len(config["texts"])+1)}
        if {(s["seed"],s["textIndex"]) for s in report["samples"]} != expected or len(report["samples"]) != len(expected):
            raise ValueError("实验组输出不完整，不能生成成功报告")
        embeddings=[]
        for sample in report["samples"]:
            style=style_for_audio(output/case/sample["file"],camp,torch)
            embeddings.append(style)
            sample["heldOutCosine"]=float(np.mean([F.cosine_similarity(style,h).item() for h in held]))
        report["meanHeldOutCosine"]=float(np.mean([s["heldOutCosine"] for s in report["samples"]]))
        report["meanCrossTextCosine"]=float(np.mean([F.cosine_similarity(a,b).item() for i,a in enumerate(embeddings) for j,b in enumerate(embeddings) if i<j and report["samples"][i]["seed"]==report["samples"][j]["seed"]]))
        results.append(report)
    make_blind(output)
    save(output/"results.json",{"configuration":config,"results":results,"subjectiveVerdict":"pending-user-listening",
         "formalFusionEnabled":False,"metricLimit":"Same CAMPPlus family as conditioning; first 15 seconds per generated sample; auxiliary only, not independent subjective quality.",
         "memoryScope":"PyTorch allocated/reserved in synthesis child only; RSS is process memory, not whole machine."})
    print(json.dumps([{k:r[k] for k in ["case","meanHeldOutCosine","meanCrossTextCosine"]} for r in results]))


def run(output):
    from indextts.material_service import write_json
    config=checked_configuration(output)
    runs=[]
    for stage,case in [("encode",None)]+[("synthesize",c) for c in config["cases"]]+[("summarize",None)]:
        command=[sys.executable,str(Path(__file__).resolve()),stage,"--output",str(output)]
        if case: command += ["--case",case]
        log=output/(f"{stage}-{case or 'all'}.log")
        began=time.perf_counter()
        print(f'Start isolated {stage} {case or ""}',flush=True)
        try:
            with log.open("w",encoding="utf-8") as stream:
                completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800,
                    env={**os.environ,"PYTHONUTF8":"1"},creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            code=completed.returncode
        except subprocess.TimeoutExpired: code="timeout"
        runs.append({"stage":stage,"case":case,"returncode":code,"seconds":time.perf_counter()-began,"log":log.name})
        write_json(output/"isolated-runs.json",runs)
        if code!=0: raise RuntimeError(f"隔离实验失败 {stage}/{case}: {code}，详见 {log}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","run","encode","synthesize","blind","summarize"])
    parser.add_argument("--output",required=True)
    parser.add_argument("--workspace",default=str(ROOT/"outputs/voice-workbench"))
    parser.add_argument("--material-id")
    parser.add_argument("--segments")
    parser.add_argument("--held-out")
    parser.add_argument("--primary")
    parser.add_argument("--confirmed-target",action="store_true")
    parser.add_argument("--confirmed-pause-target",action="store_true",help="单独确认空白对照整个区间的目标人物")
    parser.add_argument("--pause-start",type=float,default=0)
    parser.add_argument("--pause-end",type=float,default=15)
    parser.add_argument("--source-model-dir",default=str(ROOT/"voice-producer/models/checkpoints"))
    parser.add_argument("--reader-model-dir")
    parser.add_argument("--case")
    args=parser.parse_args()
    if args.stage=="prepare": prepare(args)
    elif args.stage=="run": run(Path(args.output).resolve())
    elif args.stage=="encode": encode(Path(args.output).resolve())
    elif args.stage=="synthesize": synthesize(Path(args.output).resolve(),args.case)
    elif args.stage=="blind": print(json.dumps(make_blind(Path(args.output).resolve())))
    else: summarize(Path(args.output).resolve())


if __name__=="__main__": main()
