"""Inspect current local artifacts, without constructing any synthesis model."""
from pathlib import Path
import argparse
import json,math,sys
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from indextts.voicepack.provenance import sha256_file
from indextts.voicepack.archive import load_voicepack,VoicePackError,tensor_manifest
from indextts.voicepack.schema import TENSOR_RULES,_validate_fixed_tensor_shapes
from tools.reference_fusion_experiment import checked_configuration,silence_metrics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-dir',type=Path,default=ROOT/'outputs/reference-fusion-20261001')
parser.add_argument('--record-dir',type=Path,default=ROOT/'docs/validation/long-material-2026-10-01')
parser.add_argument('--qa-voice-dir',type=Path,default=ROOT/'outputs/material-browser-qa/work/voices/material-qa-20261001')
parser.add_argument('--main-workspace',type=Path,default=ROOT/'outputs/voice-workbench')
parser.add_argument('--audit-output',type=Path,default=ROOT/'outputs/material-artifact-audit.json')
args=parser.parse_args()
run=args.experiment_dir.resolve()
doc=args.record_dir.resolve()
config=checked_configuration(run)
saved=json.loads((doc/'experiment-results.json').read_text(encoding='utf8'))
inputs=json.loads((doc/'experiment-inputs.json').read_text(encoding='utf8'))
reader_root=Path(config['readerModels'])
reader_manifest=reader_root/'runtime_model.json'
assert sha256_file(reader_manifest)==inputs['readerManifestSha256']
reader=json.loads(reader_manifest.read_text(encoding='utf8'))
assert reader['sourceModelFingerprint']==json.loads((doc/'encoding.json').read_text(encoding='utf8'))['sourceModelFingerprint']
for name,spec in reader['files'].items():
    path=(reader_root/name).resolve()
    assert path.parent==reader_root.resolve() and path.stat().st_size==spec['bytes'] and sha256_file(path)==spec['sha256']
expected={(seed,i) for seed in config['seeds'] for i in range(1,len(config['texts'])+1)}
assert len(config['cases'])==7 and len(expected)==8
assert config['generationSettings']==saved['configuration']['generationSettings']
base=load_file(str(run/'baseline.safetensors'))
styles=load_file(str(run/'styles.safetensors'))
model_root=Path(config['sourceModels'])
cfg=OmegaConf.load(model_root/'config.yaml')
# Map the checkpoint; read only the original speaker projection.
state=torch.load(model_root/cfg.gpt_checkpoint,map_location='cpu',mmap=True,weights_only=True)
state=state.get('model',state)
weight,bias=state['spk_emb_proj.weight'],state['spk_emb_proj.bias']
checks=[]
for case in config['cases']:
    path=run/(case+'.safetensors')
    assert sha256_file(path)==json.loads((doc/'encoding.json').read_text(encoding='utf8'))['tensorHashes'][path.name]
    tensors=load_file(str(path))
    assert set(tensors)==set(TENSOR_RULES)
    for name,tensor in tensors.items():
        assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all() and tensor.ndim==TENSOR_RULES[name][0]
    _validate_fixed_tensor_shapes(tensor_manifest(tensors))
    if case not in {'baseline','pause-keep','pause-short'}:
        for name in ['prompt_condition','ref_mel','base_emotion','emotion_basis']:
            assert torch.equal(tensors[name],base[name])
        method,count=case.split('-')
        selected=torch.stack([styles[sid] for sid in config['trainingIds'][:int(count)]])
        style=selected.mean(0) if method=='mean' else (selected/selected.norm(dim=-1,keepdim=True)).mean(0)*selected.norm(dim=-1,keepdim=True).mean(0)
        assert torch.equal(style,tensors['speaker_style'])
        assert torch.equal(torch.nn.functional.linear(style,weight,bias),tensors['speaker_latent'])
    result=next(r for r in saved['results'] if r['case']==case)
    assert {(s['seed'],s['textIndex']) for s in result['samples']}==expected and len(result['samples'])==8
    for sample in result['samples']:
        path=run/case/sample['file']
        assert sha256_file(path)==sample['waveformSha256']
        audio,rate=sf.read(path,dtype='float32')
        assert rate==22050 and np.isfinite(audio).all() and len(audio)>0
        actual=silence_metrics(audio,rate)
        assert all(math.isclose(actual[key],sample[key],abs_tol=1e-8) for key in actual if isinstance(actual[key],(int,float)))
    try:load_voicepack(run/(case+'.safetensors'))
    except VoicePackError:pass
    else:raise AssertionError('Experimental artifact accepted as a formal IVP')
    checks.append({'case':case,'validSamples':8,'rawTensorHashMatches':True,'finiteFp32SixTensorAbi':True,
        'fixedNonIdentityTensors':not case.startswith('pause-'),'identityFormulaAndOriginalProjection':case.startswith(('mean-','normalized-')),
        'rejectedByFormalPackLoader':True})

for path in (run/'clips').glob('*.wav'):
    assert sf.info(path).duration<=15+1/22050
keep,rate=sf.read(run/'clips/pause-keep.wav',dtype='float32')
short,other_rate=sf.read(run/'clips/pause-short.wav',dtype='float32')
parts=[];start=0
for a,b in config['recipe']['removedGaps']:
    parts.append(keep[start:a]);start=b
parts.append(keep[start:])
assert rate==other_rate and np.array_equal(np.concatenate(parts),short)
assert set(config['trainingIds']).isdisjoint(config['heldOutIds'])
key=json.loads((run/'blind-key.json').read_text(encoding='utf8'))
blind_groups={'identity':{label:case for label,case in key.items() if label.startswith('V')},
              'pause':{label:case for label,case in key.items() if label.startswith('P')}}
for group,mapping in blind_groups.items():
    assert set(mapping.values())=={c for c in config['cases'] if c.startswith('pause-')==(group=='pause')}
    for seed,index in expected:
        stem=f'{group}-seed-{seed}-text-{index}'
        info=json.loads((run/'blind'/(stem+'.json')).read_text(encoding='utf8'))
        assert [s['label'] for s in info]==list(mapping)
        originals=[sf.read(run/case/f'seed-{seed}-text-{index}.wav',dtype='float32')[0] for case in mapping.values()]
        rms=min(float(np.sqrt(np.mean(w**2))) for w in originals)
        offset=0.;pieces=[]
        for entry,(label,case),wave in zip(info,mapping.items(),originals):
            gain=min(1.,rms/max(1e-8,float(np.sqrt(np.mean(wave**2)))))
            expected_audio=wave*gain
            actual,rate=sf.read(run/'blind'/f'{label}-seed-{seed}-text-{index}.wav',dtype='float32')
            assert rate==22050 and actual.shape==expected_audio.shape and np.max(np.abs(actual-expected_audio))<=1/32768+1e-6
            assert math.isclose(entry['startSeconds'],offset,abs_tol=1e-8)
            assert math.isclose(entry['durationSeconds'],len(wave)/rate,abs_tol=1e-8)
            pieces.extend([expected_audio,np.zeros(rate,dtype=np.float32)])
            offset+=len(wave)/rate+1
        joined=np.concatenate(pieces)
        actual,rate=sf.read(run/'blind'/(stem+'.wav'),dtype='float32')
        assert rate==22050 and actual.shape==joined.shape and np.max(np.abs(actual-joined))<=1/32768+1e-6
isolated=json.loads((run/'isolated-runs.json').read_text(encoding='utf8'))
assert len(isolated)==9 and all(s['returncode']==0 for s in isolated)
browser=json.loads((doc/'browser-end-to-end.json').read_text(encoding='utf8'))
library=args.qa_voice_dir.resolve()
for profile,data in browser['profiles'].items():
    path=library/(profile+'.ivp');pack=load_voicepack(path)
    assert sha256_file(path)==data['packSha256'] and pack.manifest==data['manifest']
    latest=json.loads((library/(profile+'-preview')/'latest.json').read_text(encoding='utf8'))
    assert sha256_file(library/(profile+'-preview')/latest['wav'])==data['wavSha256']
assert not (args.main_workspace.resolve()/'voices'/library.name/'voice.json').exists()
output={'artifactAuditPassed':True,'cases':checks,'validSamples':56,'isolatedStages':9,
    'trainingHeldoutDisjoint':True,'allEncodedInputClipsAtMost15Seconds':True,
    'pauseShorteningPreservesOtherSamplesExactly':True,'bothFormalProfilesMatchSavedBrowserEvidence':True,
    'experimentalArtifactsRejectedByFormalPackLoader':True,'mainVoiceLibraryHasNoQaEntries':True,
    'readerModelFilesMatchRecordedManifest':True,
    'anonymousSamplesAndTimingIndexesMatchOriginals':True,
    'sourceModelFingerprint':json.loads((doc/'encoding.json').read_text(encoding='utf8'))['sourceModelFingerprint'],
    'qualityGatePassed':False,'qualityLimit':'User identified BGM in primary reference; overall fusion benefit not confirmed. Technical execution evidence is not clean-reference quality acceptance.',
    'formalFusionEnabled':False}
args.audit_output.resolve().write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(output,ensure_ascii=False))
