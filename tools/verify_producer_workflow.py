"""Real GPU acceptance; isolated drafts, no writes to the installed voice library."""
import json
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from indextts.producer_drafts import ProducerDrafts
from indextts.workbench_service import WorkbenchService
from indextts.runtime.profiles import PROFILES
from indextts.voicepack.archive import load_voicepack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-id',required=True,help='Existing local material to copy into isolated acceptance drafts')
    args=parser.parse_args()
    output=ROOT/'outputs/producer-dual-acceptance'
    output.mkdir(parents=True,exist_ok=True)
    service=WorkbenchService(output/'jobs',ROOT/'outputs/voice-workbench',modules=('producer','audition'))
    drafts=ProducerDrafts(output/'drafts',service,ROOT/'voice-producer/models/checkpoints',ROOT/'voice-producer/models/runtime')
    records=[]
    try:
        d=drafts.new();wid=d['workspaceId']
        d=drafts.import_legacy(wid,args.source_id)
        candidates=[r for r in d['editor']['rows'] if r[3] and r[2]-r[1] >= 4][:2]
        if len(candidates)<2:raise ValueError('Acceptance requires two selected clips of at least four seconds')
        rows=[[f'accept-{i}',r[1],min(r[2],r[1]+5),True] for i,r in enumerate(candidates)]
        e={'sourceId':d['sourceId'],'rows':rows,'primary':rows[0][0],
           'previewStarts':{rows[0][0]:rows[0][1]+1},'editing':rows[0]}
        def save(**changes):
            nonlocal d
            d=drafts.save(wid,dict(workspaceId=wid,revision=d['revision'],clientId='acceptance',
                sequence=d['clients'].get('acceptance',0)+1,**changes))
        save(editor=e,name='双精度流程验证',gender='female',text='你好。')
        print('BUILD FP32 + BF16',flush=True)
        built,d=drafts.run(wid,'build')
        assert len(load_voicepack(drafts.directory(wid)/built['file']).available_profiles)==2
        for profile in PROFILES:
            save(profile=profile)
            print('AUDITION',profile,flush=True)
            preview,d=drafts.run(wid,'audition')
            assert preview['version']==built['version']
            assert d['previews'][0]['profile']==profile and len(d['packages'])==1
            records.append(dict(profile=profile,workspaceId=wid,voiceId=d['voiceId'],packageVersion=built['version'],
                                preview=d['previews'][0]))
            print('PASS AUDITION',profile,flush=True)
        save(name='双精度下载验证',gender='male')
        exported,d=drafts.run(wid,'export')
        download=Path(drafts.export_path(wid,exported))
        shutil.copyfile(download,output/'dual-voice.ivp')
        for profile in PROFILES:
            pack=load_voicepack(download,profile=profile)
            assert pack.manifest['displayName']==d['name'] and pack.manifest['gender']=='male'
            assert pack.manifest['provenance']['profile']==profile
            with zipfile.ZipFile(drafts.directory(wid)/built['file']) as a,zipfile.ZipFile(download) as b:
                assert a.read('conditioning.safetensors')==b.read('conditioning.safetensors')
                assert a.read('bf16/conditioning.safetensors')==b.read('bf16/conditioning.safetensors')
                assert len(b.namelist())==7 and not any(n.endswith('.wav') for n in b.namelist())
        restored=ProducerDrafts(drafts.root,service,drafts.source_model_dir,drafts.runtime_root).activate(wid)
        assert restored['editor']==d['editor'] and restored['previews']==d['previews']
        assert len(d['packages'])==2
        (output/'results.json').write_text(json.dumps(dict(records=records,download=str(download),schema=4,
            manifest=pack.container_manifest),ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS DUAL EXPORT + RESTART',flush=True)
    finally:
        drafts.close();service.close()


if __name__=='__main__':main()
