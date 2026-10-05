import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import uuid
import zipfile

import numpy as np
import pytest
import soundfile as sf

from indextts.material_service import MaterialService, write_json
from indextts.material_web import table_rows
from indextts.producer_drafts import ProducerDrafts, reference_key, package_key, repackage
from indextts.voicepack.archive import write_voicepack, load_voicepack
from indextts.voicepack.provenance import sha256_file
from test_voicepack import _manifest, _tensors, _licenses


@pytest.fixture
def drafts(tmp_path):
    legacy=MaterialService(tmp_path/'legacy')
    source=uuid.uuid4().hex; directory=legacy.directory(source); directory.mkdir()
    sf.write(directory/'audio.wav',np.zeros(22050*12),22050)
    record=dict(sourceId=source,name='source',sourceSha256='a'*64,
        audioSha256=sha256_file(directory/'audio.wav'),durationSeconds=12,speechSeconds=10,
        primary='a',segments=[dict(id='a',start=1,end=5,selected=True),dict(id='b',start=6,end=10,selected=False)])
    record['revision']=legacy.selection_revision(record); write_json(directory/'material.json',record)
    output=tmp_path/'jobs'; output.mkdir()
    calls=[]
    def build(selection,vid,name,gender,profile,*args,**kwargs):
        calls.append(copy.deepcopy(selection))
        job=output/uuid.uuid4().hex; job.mkdir()
        from indextts.voicepack.provenance import PREPROCESS_FINGERPRINT
        manifest=_manifest(); manifest.update(voiceId=vid,displayName=name,gender=gender,schemaVersion=2,
            provenance=dict(profile=profile,referenceSha256='b'*64,referenceEncoderFingerprint='c'*64,
                            preprocessFingerprint=PREPROCESS_FINGERPRINT,producerVersions={}))
        tensors=_tensors()
        if profile=='compatible-fp32':tensors={n:t.float() for n,t in tensors.items()}
        return str(write_voicepack(job/'voice.ivp',manifest,tensors,_licenses())),{}
    audio=tmp_path/'result.wav'; sf.write(audio,np.zeros(2205),22050)
    def synth(*args):
        sf.write(audio,np.zeros(2205),22050)
        return str(audio),{'text':args[5]}
    wb=SimpleNamespace(materials=legacy,output_dir=output,build_selection=build,
        gpu=SimpleNamespace(use=lambda *a:nullcontext()),audition_service=SimpleNamespace(synthesize=synth))
    service=ProducerDrafts(tmp_path/'drafts',wb,tmp_path/'models',tmp_path/'runtime')
    d=service.new(); d=service.import_legacy(d['workspaceId'],source)
    service.runtime=lambda profile:tmp_path/'runtime'
    return service,d,calls


def save(service,d,**changes):
    payload=dict(workspaceId=d['workspaceId'],revision=d['revision'],clientId='test',sequence=d['clients'].get('test',0)+1,**changes)
    return service.save(d['workspaceId'],payload)


def test_recovery_switch_and_trash_isolation(drafts):
    s,d,_=drafts; wid=d['workspaceId']
    e=copy.deepcopy(d['editor']); e['previewStarts']={'a':2.3}; e['editing']=e['rows'][0]
    d=save(s,d,name='first',editor=e,filters=dict(silence=.8,minimum=2,enabled=True,volume=-35))
    second=s.new(); second=save(s,second,name='second')
    assert s.activate(wid)['editor']==e
    reloaded=ProducerDrafts(s.root,s.workbench,s.source_model_dir,s.runtime_root)
    assert reloaded.current()['workspaceId']==wid
    empty=s.clear(wid)
    assert empty['sourceId'] is None and len(s.items(recycled=True))==1
    assert s.read(second['workspaceId'])['name']=='second'
    assert s.workbench.materials.record(d['sourceId'])['sourceId']==d['sourceId']
    restored=s.restore(wid)
    assert restored['editor']==e and restored['voiceId']==d['voiceId']
    with pytest.raises(ValueError):s.directory('../models')


def test_revision_guards_and_late_write(drafts):
    s,d,_=drafts; old=copy.deepcopy(d)
    d=save(s,d,name='saved')
    with pytest.raises(ValueError,match='其他页面'):
        s.save(d['workspaceId'],dict(workspaceId=d['workspaceId'],revision=old['revision'],clientId='another',sequence=1,name='wrong'))
    s.save(d['workspaceId'],dict(workspaceId=d['workspaceId'],revision=old['revision'],clientId='test',sequence=0,name='late'))
    assert s.read(d['workspaceId'])['name']=='saved'
    s.clear(d['workspaceId'])
    with pytest.raises(OSError):save(s,d,name='resurrect')


def test_reuse_metadata_and_reference_rebuild(drafts):
    s,d,calls=drafts; d=save(s,d,name='voice')
    first,d=s.run(d['workspaceId'],'build')
    assert len(calls)==2 and calls[0]['method']=='primary-only-v1'
    d=save(s,d,text='new text')
    reused,d=s.run(d['workspaceId'],'export')
    assert first==reused and len(calls)==2
    d=save(s,d,name='renamed',gender='male')
    renamed,d=s.run(d['workspaceId'],'export')
    assert len(calls)==2 and renamed['version']!=first['version']
    with zipfile.ZipFile(s.directory(d['workspaceId'])/first['file']) as a,zipfile.ZipFile(s.directory(d['workspaceId'])/renamed['file']) as b:
        for name in a.namelist():
            if not name.endswith('manifest.json'):assert a.read(name)==b.read(name)
    pack=load_voicepack(s.directory(d['workspaceId'])/renamed['file'])
    assert pack.manifest['displayName']=='renamed' and pack.voice_id==d['voiceId']
    exported=s.export_path(d['workspaceId'],renamed)
    assert Path(exported).name=='音色-renamed.ivp'
    assert load_voicepack(exported).manifest==pack.manifest
    e=copy.deepcopy(d['editor']); e['rows'][1][3]=True; e['primary']='b'
    d=save(s,d,editor=e)
    second,d=s.run(d['workspaceId'],'build')
    assert len(calls)==4 and calls[-1]['method']=='speaker-mean-v1' and calls[-1]['primary']=='b'
    with pytest.raises(ValueError,match='修改'):s.export_path(d['workspaceId'],renamed)
    d=save(s,d,profile='fixed-voice-bf16')
    same,d=s.run(d['workspaceId'],'build'); assert len(calls)==4 and same['version']==second['version']


def test_task_edit_and_cancel_never_overwrite_draft(drafts):
    s,d,calls=drafts; d=save(s,d,name='voice')
    with s.task(d['workspaceId']) as (snapshot,cancel):
        newer=save(s,d,name='edited during job')
        entry=s.ensure_package(snapshot,cancel)
        assert s.read(d['workspaceId'])['name']==newer['name']
        assert s.read(d['workspaceId'])['currentPackage'] is None
    original=s.workbench.build_selection
    def cancelled_build(*args,**kwargs):
        result=original(*args,**kwargs); kwargs['cancelled'].set(); return result
    s.workbench.build_selection=cancelled_build
    e=copy.deepcopy(newer['editor']);e['rows'][0][1]=1.2
    newer=save(s,newer,editor=e)
    with pytest.raises(RuntimeError,match='取消'):s.run(d['workspaceId'],'export')
    assert len(s.read(d['workspaceId'])['packages'])==1


def test_enabled_references_retention_primary_fallback_and_reuse(drafts):
    s,d,calls=drafts
    editor=copy.deepcopy(d['editor']); editor['rows'][1][3]=True
    editor['previewStarts']={'a':2.3}; editor['editing']=editor['rows'][0].copy()
    d=save(s,d,name='voice',editor=editor)
    original_key=reference_key(d)
    first,d=s.run(d['workspaceId'],'build')
    assert calls[-1]['method']=='speaker-mean-v1' and calls[-1]['primary']=='a'
    editor=copy.deepcopy(d['editor']); editor['disabled']=['a']
    d=save(s,d,editor=editor)
    assert all(r[3] for r in d['editor']['rows']) and d['editor']['primary']=='a'
    assert d['editor']['previewStarts']=={'a':2.3}
    disabled_key=reference_key(d); disabled_revision=d['referenceRevision']
    second,d=s.run(d['workspaceId'],'export')
    assert second['version']!=first['version'] and calls[-1]['method']=='primary-only-v1'
    assert calls[-1]['primary']=='b' and [r['id'] for r in calls[-1]['segments']]==['b']
    # Editing a retained, disabled clip changes the draft only.
    editor=copy.deepcopy(d['editor']); editor['rows'][0][1]=1.2
    d=save(s,d,editor=editor)
    assert reference_key(d)==disabled_key and d['referenceRevision']==disabled_revision
    s.new(); reloaded=ProducerDrafts(s.root,s.workbench,s.source_model_dir,s.runtime_root)
    restored=reloaded.activate(d['workspaceId'])
    assert restored['editor']==editor
    cleared=s.clear(d['workspaceId']); restored=s.restore(d['workspaceId'])
    assert restored['editor']==editor and cleared['workspaceId']!=restored['workspaceId']
    editor['rows'][0][1]=1; editor['disabled']=[]
    d=save(s,restored,editor=editor)
    assert reference_key(d)==original_key
    reused,d=s.run(d['workspaceId'],'export')
    assert reused['version']==first['version'] and len(calls)==4


def test_all_disabled_rejects_build_audition_export_without_losing_results(drafts):
    s,d,calls=drafts; d=save(s,d,name='voice')
    first,d=s.run(d['workspaceId'],'audition'); previous=copy.deepcopy(d['previews'])
    editor=copy.deepcopy(d['editor']); editor['disabled']=['a']; d=save(s,d,editor=editor)
    for operation in ('build','audition','export'):
        with pytest.raises(ValueError,match='至少启用一个'):
            s.run(d['workspaceId'],operation)
    current=s.read(d['workspaceId'])
    assert current['editor']==editor and current['currentPackage']==first
    assert current['previews']==previous and len(calls)==2
    with pytest.raises(ValueError,match='修改'):
        s.export_path(d['workspaceId'],first)


@pytest.mark.parametrize('disabled', [['missing'],['b'],['a','a'],'a',None,[{}]])
def test_invalid_activation_state_is_not_saved(drafts,disabled):
    s,d,_=drafts; editor=copy.deepcopy(d['editor']); editor['disabled']=disabled
    with pytest.raises(ValueError,match='启用状态'):
        save(s,d,editor=editor)
    assert s.read(d['workspaceId'])==d


def test_activation_edit_during_build_preserves_newer_draft(drafts):
    s,d,_=drafts; d=save(s,d,name='voice')
    original=s.workbench.build_selection
    def edited_build(*args,**kwargs):
        result=original(*args,**kwargs)
        current=s.read(d['workspaceId']); editor=copy.deepcopy(current['editor'])
        editor['disabled']=['a']; save(s,current,editor=editor)
        return result
    s.workbench.build_selection=edited_build
    entry,current=s.run(d['workspaceId'],'build')
    assert current['editor']['disabled']==['a'] and current['currentPackage'] is None
    assert entry['editor'].get('disabled',[])==[]


def test_preview_rotation_failure_and_binding(drafts):
    s,d,calls=drafts;d=save(s,d,name='voice')
    _,d=s.run(d['workspaceId'],'audition');first=copy.deepcopy(d['previews'][0])
    d=save(s,d,text='second')
    _,d=s.run(d['workspaceId'],'audition')
    assert d['previews'][1]==first and d['previews'][0]['text']=='second' and len(calls)==2
    before=copy.deepcopy(d['previews'])
    s.workbench.audition_service.synthesize=lambda *a:(_ for _ in ()).throw(RuntimeError('failed'))
    with pytest.raises(RuntimeError):s.run(d['workspaceId'],'audition')
    assert s.read(d['workspaceId'])['previews']==before


def test_manual_emotion_recovery_package_reuse_and_task_snapshot(drafts):
    s,d,calls=drafts; d=save(s,d,name='voice')
    entry,d=s.run(d['workspaceId'],'build')
    vector=[.45,0,0,0,0,0,.2,0]
    original_ref=d['referenceRevision']; original_key=package_key(d)
    d=save(s,d,emotion=vector)
    assert package_key(d)==original_key and d['referenceRevision']==original_ref
    second=s.new()
    assert second['emotion']=='base' and s.activate(d['workspaceId'])['emotion']==vector
    original=s.workbench.audition_service.synthesize
    received=[]
    def synth(*args):
        received.append(copy.deepcopy(args[6]))
        save(s,s.read(d['workspaceId']),emotion=[0,0,.6,0,0,0,0,0])
        return original(*args)
    s.workbench.audition_service.synthesize=synth
    reused,d=s.run(d['workspaceId'],'audition')
    assert reused['version']==entry['version'] and len(calls)==2
    assert received==[vector] and d['previews'][0]['emotion']==vector
    assert d['emotion']==[0,0,.6,0,0,0,0,0]
    d=save(s,d,emotion=[0]*8)
    assert d['emotion']=='base'


@pytest.mark.parametrize('vector', [[0]*7,[0]*9,[True]+[0]*7,[float('nan')]+[0]*7,
                                   [float('inf')]+[0]*7,[-.1]+[0]*7,[1.21]+[0]*7,'manual'])
def test_invalid_emotion_preserves_saved_draft(drafts,vector):
    s,d,_=drafts
    with pytest.raises(ValueError,match='情感向量'):
        save(s,d,emotion=vector)
    assert s.read(d['workspaceId'])==d


def test_legacy_draft_defaults_to_base_emotion(drafts):
    s,d,_=drafts
    d.pop('emotion')
    write_json(s.directory(d['workspaceId'])/'draft.json',d)
    assert s.activate(d['workspaceId'])['emotion']=='base'


def test_failed_build_keeps_package_and_export_rejects_concurrent_edit(drafts):
    s,d,_=drafts;d=save(s,d,name='voice')
    first,d=s.run(d['workspaceId'],'build')
    e=copy.deepcopy(d['editor']);e['rows'][0][1]=1.1;d=save(s,d,editor=e)
    build=s.workbench.build_selection
    s.workbench.build_selection=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('encoding failed'))
    with pytest.raises(RuntimeError,match='encoding failed'):s.run(d['workspaceId'],'export')
    assert s.read(d['workspaceId'])['currentPackage']==first
    assert len(s.read(d['workspaceId'])['packages'])==1
    def concurrent(*a,**k):
        result=build(*a,**k)
        save(s,s.read(d['workspaceId']),name='newer draft')
        return result
    s.workbench.build_selection=concurrent
    with pytest.raises(ValueError,match='制作期间'):s.run(d['workspaceId'],'export')
    assert s.read(d['workspaceId'])['name']=='newer draft'
    assert s.read(d['workspaceId'])['currentPackage']==first


def test_save_failure_preserves_previous_atomic_record(drafts,monkeypatch):
    s,d,_=drafts;d=save(s,d,name='safe')
    monkeypatch.setattr('indextts.producer_drafts.atomic_json',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError,match='disk full'):save(s,d,name='lost write')
    assert s.read(d['workspaceId'])==d


def test_workflow_ui_hides_legacy_tools_and_keeps_legacy_mode(tmp_path):
    from indextts.validation_web import create_app
    app=create_app(modules=('producer','audition'),producer_workflow=True,
                   workspace_dir=tmp_path/'library',drafts_dir=tmp_path/'drafts',output_dir=tmp_path/'jobs')
    labels=[c['props'].get('label') or c['props'].get('value') for c in app.config['components']]
    assert '生成音色包' in labels and '制作并进入试听' not in labels
    assert '本次试听' in labels and '上次试听' not in labels
    assert '试听精度' in labels and '制包精度' not in labels
    precision=next(c for c in app.config['components'] if c['props'].get('label')=='试听精度')
    area=next(c for c in app.config['components'] if c['props'].get('elem_id')=='producer-audition')
    def contained(node,target):
        return node.get('id')==target or any(contained(child,target) for child in node.get('children',[]))
    area_layout=next(node for node in _layout_nodes(app.config['layout']) if node['id']==area['id'])
    assert contained(area_layout,precision['id'])
    assert '情感向量（手动设置）' in labels
    assert {'高兴','愤怒','悲伤','恐惧','厌恶','低落','惊讶','平静'} <= set(v for v in labels if isinstance(v,str))
    assert '参考构建方法' not in labels and '音色库与下载' not in labels and '诊断与模型' not in labels
    assert all(c['props'].get('interactive') for c in app.config['components']
               if c['props'].get('label') in {'音色名称','试听文本','分段停顿（秒）','最短片段（秒）'})
    apis={d['api_name'] for d in app.config['dependencies']}
    assert {'workspace_list','workspace_new','workspace_load','workspace_save','workspace_clear',
            'workspace_restore','workspace_build','workspace_audition','workspace_cancel','workspace_export'} <= apis
    app.workbench.close()


def _layout_nodes(node):
    yield node
    for child in node.get('children',[]):yield from _layout_nodes(child)


def test_profile_switch_reuses_dual_package_and_binds_audition(drafts):
    s,d,calls=drafts; d=save(s,d,name='dual')
    first,d=s.run(d['workspaceId'],'audition')
    before=d['referenceRevision']; key=package_key(d)
    assert d['previews'][0]['profile']=='compatible-fp32'
    assert load_voicepack(s.directory(d['workspaceId'])/first['file']).available_profiles==('compatible-fp32','fixed-voice-bf16')
    d=save(s,d,profile='fixed-voice-bf16')
    assert package_key(d)==key and d['referenceRevision']==before
    reused,d=s.run(d['workspaceId'],'audition')
    assert reused['version']==first['version'] and len(calls)==2
    assert d['previews'][0]['profile']=='fixed-voice-bf16'
    assert d['previews'][1]['profile']=='compatible-fp32'
    with zipfile.ZipFile(s.directory(d['workspaceId'])/reused['file']) as archive:
        assert json.loads(archive.read('manifest.json'))['schemaVersion']==4


def test_second_precision_failure_preserves_successful_dual_package(drafts):
    s,d,calls=drafts; d=save(s,d,name='dual')
    first,d=s.run(d['workspaceId'],'build')
    editor=copy.deepcopy(d['editor']); editor['rows'][0][1]=1.1; d=save(s,d,editor=editor)
    original=s.workbench.build_selection
    def failed(*args,**kwargs):
        if args[4]=='fixed-voice-bf16':raise RuntimeError('BF16 failed')
        return original(*args,**kwargs)
    s.workbench.build_selection=failed
    with pytest.raises(RuntimeError,match='BF16 failed'):s.run(d['workspaceId'],'export')
    restored=s.read(d['workspaceId'])
    assert restored['currentPackage']==first and len(restored['packages'])==1
    assert len(list((s.directory(d['workspaceId'])/'packages').glob('*.ivp')))==1
    assert not list(s.workbench.output_dir.iterdir())


def test_legacy_profile_cache_upgrades_without_reencoding(drafts):
    s,d,calls=drafts; d=save(s,d,name='dual')
    first,d=s.run(d['workspaceId'],'build')
    from indextts.voicepack.archive import extract_voicepack
    old=[]
    for profile in ('compatible-fp32','fixed-voice-bf16'):
        path=s.directory(d['workspaceId'])/'packages'/(profile+'.ivp')
        extract_voicepack(s.directory(d['workspaceId'])/first['file'],path,profile)
        old.append(dict(first,file='packages/'+path.name,profile=profile,
                        referenceKey=reference_key(d,legacy_profile=profile),key='old-'+profile))
    d['packages']=old;d['currentPackage']=old[0]
    write_json(s.directory(d['workspaceId'])/'draft.json',d)
    upgraded,d=s.run(d['workspaceId'],'build')
    assert len(calls)==2 and upgraded['version']!=first['version']
    assert len(load_voicepack(s.directory(d['workspaceId'])/upgraded['file']).available_profiles)==2


def test_package_buttons_do_not_synthesize_or_reload_preview(drafts):
    import gradio as gr
    from indextts.producer_workflow_web import build_page
    s,d,_=drafts; d=save(s,d,name='voice',text='first preview')
    _,d=s.run(d['workspaceId'],'audition')
    previous=copy.deepcopy(d['previews'])
    synth=s.workbench.audition_service.synthesize; synthesized=[]
    def tracked(*args):
        synthesized.append(args[5]); return synth(*args)
    s.workbench.audition_service.synthesize=tracked
    with gr.Blocks() as app:
        build_page(app,s)
    callbacks={fn.api_name:fn.fn for fn in app.fns.values()}
    def click(kind,**changes):
        current=s.read(d['workspaceId'])
        payload=dict(workspaceId=d['workspaceId'],revision=current['revision'],clientId='test',
                     sequence=current['clients']['test']+1,**changes)
        return callbacks['workspace_'+kind](d['workspaceId'],json.dumps(payload),progress=None)
    # A reference edit forces real rebuilding through the fixture builder.
    editor=copy.deepcopy(d['editor']); editor['rows'][0][1]=1.1
    values=click('build',editor=editor)
    assert values[3]==gr.skip() and '修改前的试听' in values[2]
    assert s.read(d['workspaceId'])['previews']==previous and not synthesized
    values=click('export',name='renamed')
    assert values[3]==gr.skip() and values[4].endswith('.ivp')
    assert s.read(d['workspaceId'])['previews']==previous and not synthesized
    values=click('audition',text='explicit audition')
    assert isinstance(values[3],str) and Path(values[3]).exists()
    assert synthesized==['explicit audition']
