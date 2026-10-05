"""Single-page voice authoring workflow; legacy validation pages remain separate."""
import copy
import json
from datetime import datetime
from pathlib import Path
import gradio as gr
from .material_player import player_html, INITIALIZE
from .producer_drafts import package_key, reference_key
from .runtime.profiles import FP32, BF16

INITIALIZE_WORKFLOW = Path(__file__).with_name('producer_workflow.js').read_text(encoding='utf-8')
WORKFLOW_CSS = Path(__file__).with_name('producer_workflow.css').read_text(encoding='utf-8')
EMOTION_LABELS = ('高兴', '愤怒', '悲伤', '恐惧', '厌恶', '低落', '惊讶', '平静')
FLUSH = "(...args) => { const w=document._producerWorkflow; if(w){clearTimeout(w.timer); args[1]=w.collect();} return args.slice(0,2); }"
FLUSH_THREE = "(...args) => { const w=document._producerWorkflow; if(w){clearTimeout(w.timer); args[2]=w.collect();} return args.slice(0,3); }"
ACK = """(ack) => {const a=JSON.parse(ack||'{}'),w=document._producerWorkflow;
  if(w&&w.context.workspaceId===a.workspaceId){w.context.revision=a.revision;
    const key=`producer-pending:${a.workspaceId}`,p=JSON.parse(localStorage.getItem(key)||'null');
    if(p&&p.clientId===a.clientId&&p.sequence<=a.sequence)localStorage.removeItem(key);}
  return [];}"""


def local_time(value, pattern='%H:%M:%S'):
    return datetime.fromisoformat(value).astimezone().strftime(pattern)


def build_page(app, drafts):
    wid = gr.State(None)
    context = gr.Textbox(elem_id='producer-context', show_label=False)
    payload = gr.Textbox(elem_id='producer-save-payload', show_label=False)
    ack = gr.Textbox(elem_id='producer-save-ack', show_label=False)
    with gr.Column(elem_id='producer-page'):
        with gr.Row(elem_id='producer-heading'):
            gr.Markdown('# 音色制作\n选取参考 → 制作 → 试听 → 下载', elem_id='producer-title')
            save_status = gr.Markdown('正在加载工作区…', elem_id='producer-save-status')
        with gr.Row(elem_id='producer-toolbar'):
            selector = gr.Dropdown(label='工作区', interactive=True, choices=[], scale=3, min_width=210)
            new = gr.Button('新建工作区', size='sm', scale=0, min_width=100)
            last = gr.Button('恢复上次工作区', size='sm', scale=0, min_width=130)
            clear = gr.Button('清空当前工作区', size='sm', scale=0, min_width=130)
            with gr.Column(scale=0, min_width=180):
                with gr.Accordion('回收区 · 保留 7 天', open=False):
                    recycled = gr.Dropdown(label='回收工作区', choices=[], interactive=True)
                    restore = gr.Button('恢复这个工作区', size='sm')
        with gr.Row(visible=False) as clear_confirm:
            gr.Markdown('当前工作区将移入回收区，7 天内可恢复。')
            confirm = gr.Button('确认清空', variant='stop', scale=0)
            dismiss = gr.Button('取消清空', scale=0)
        with gr.Row(elem_id='producer-workspace'):
            with gr.Column(scale=2, min_width=640, elem_id='producer-reference'):
                gr.Markdown('### ① 参考素材')
                with gr.Row(elem_id='producer-source'):
                    media = gr.File(label='上传音频或视频', type='filepath', file_types=['audio','video'], height=76, scale=2)
                    with gr.Column(scale=1, min_width=185):
                        with gr.Accordion('使用已有素材', open=False):
                            legacy = gr.Dropdown(label='已有素材', choices=[(r['name'],r['sourceId']) for r in drafts.workbench.materials.items()], interactive=True)
                            import_legacy = gr.Button('复制素材到当前工作区', size='sm')
                        gr.Markdown('调整参数实时选片；加入后点击卡片微调。', elem_classes=['producer-tip'])
                with gr.Row(elem_id='producer-filters'):
                    silence = gr.Slider(.1,3,value=.5,step=.05,label='分段停顿（秒）',elem_id='material-silence',interactive=True,min_width=160)
                    minimum = gr.Slider(.1,15,value=1,step=.1,label='最短片段（秒）',elem_id='material-minimum',interactive=True,min_width=160)
                    with gr.Column(min_width=190):
                        enabled = gr.Checkbox(label='过滤小声杂音',value=False,elem_id='material-volume-enabled')
                        volume = gr.Slider(-80,0,value=-50,step=1,label='最低音量（dBFS）',elem_id='material-volume',interactive=False)
                player = gr.HTML(player_html(),show_label=False)
                editor = gr.Textbox(elem_id='material-waveform-draft',show_label=False)
            with gr.Column(scale=1, min_width=340, elem_id='producer-finish'):
                gr.Markdown('### ④ 制作音色')
                with gr.Row():
                    name = gr.Textbox(label='音色名称',placeholder='给这个声音起个名字',elem_id='producer-name',interactive=True,scale=2,min_width=150)
                    gender = gr.Dropdown([('未知','unknown'),('男声','male'),('女声','female')],value='unknown',label='性别',elem_id='producer-gender',interactive=True,scale=1,min_width=95)
                with gr.Accordion('高级设置',open=False):
                    with gr.Row():
                        device = gr.Dropdown([('自动','auto'),('CPU','cpu'),'cuda:0','cuda:1'],value='auto',label='计算设备',elem_id='producer-device',interactive=True,min_width=120)
                    gr.Markdown('一段使用单段参考；多段等权融合音色身份，主参考提供声学提示。')
                gr.Markdown('参考应为同一人，尽量避开其他人声、杂音和背景音乐。', elem_classes=['producer-tip'])
                gr.Markdown('音色包同时包含 FP32、BF16，可按设备切换使用。', elem_classes=['producer-tip'])
                with gr.Row():
                    make = gr.Button('生成音色包',variant='primary',size='sm',min_width=160)
                    cancel = gr.Button('取消',size='sm',scale=0,min_width=60)
                result = gr.Markdown('尚未制作临时音色。', elem_id='producer-result')
                with gr.Column(visible=False, min_width=0, elem_id='producer-audition') as audition_area:
                    gr.Markdown('### ⑤ 文本试听')
                    text = gr.Textbox(label='试听文本',placeholder='输入想听的内容',lines=2,elem_id='producer-text',interactive=True)
                    profile = gr.Radio([('FP32',FP32),('BF16',BF16)],value=FP32,label='试听精度',elem_id='producer-profile',interactive=True)
                    emotion_controls = []
                    with gr.Accordion('情感向量（手动设置）', open=False, elem_id='producer-emotion'):
                        gr.Markdown('可混合调整，范围 0～1.2；全部为 0 时沿用参考的基础情感。', elem_classes=['producer-tip'])
                        for i in range(0, 8, 2):
                            with gr.Row():
                                for j in (i, i+1):
                                    emotion_controls.append(gr.Slider(0,1.2,value=0,step=.05,label=EMOTION_LABELS[j],
                                        interactive=True,min_width=120,elem_id=f'producer-emotion-{j}'))
                        reset_emotion = gr.Button('全部归零',size='sm')
                    audition = gr.Button('生成试听',variant='primary',size='sm')
                    with gr.Column(min_width=0, elem_classes=['producer-preview']):
                        current_info = gr.Markdown('本次试听：尚无结果')
                        current_audio = gr.Audio(label='本次试听',show_label=False,interactive=False,type='filepath',elem_classes=['producer-audio'])
                with gr.Column(min_width=0, elem_id='producer-final-download'):
                    gr.Markdown('### ⑥ 下载定稿')
                    export = gr.Button('下载音色包',variant='primary',size='sm',elem_id='producer-export')
                    gr.Markdown('下载包含 FP32、BF16；自动更新参考和音色信息，无需先试听。', elem_classes=['producer-tip'])
                    download = gr.File(label='最新音色包（.ivp）',show_label=False,interactive=False,height=58,elem_id='producer-download-file')
    enabled.input(lambda e:gr.update(interactive=e),enabled,volume,queue=False)
    reset_emotion.click(lambda:[0]*8,outputs=emotion_controls,queue=False).then(
        None,js="()=>{document._producerWorkflow?.schedule();return [];}",queue=False)
    player.change(None,js=INITIALIZE,queue=False)
    download.change(None,download,None,queue=False,js="""(value)=>{const w=document._producerWorkflow;
      if(value&&w&&w.exportSignature===w.signature(JSON.parse(w.collect())))
        document.querySelector('#producer-download-file')?.classList.remove('draft-dirty');return [];}""")

    def choices(recycled=False):
        return [(d['name'] or ('未命名 · '+local_time(d['createdAt'],'%Y-%m-%d %H:%M:%S')),d['workspaceId']) for d in drafts.items(recycled=recycled)]

    def preview_values(d):
        values=[]
        for index,title in enumerate(('本次试听',)):
            if index >= len(d['previews']):
                values += [title+'：尚无结果',None]
            else:
                p=d['previews'][index]
                emotion = p.get('emotion', 'base')
                preview_profile=p.get('profile') or next((v.get('profile') for v in d['packages'] if v['version']==p['version']),FP32)
                stale = p['packageKey'] != package_key(d) or p['text'] != d['text'] or emotion != d.get('emotion', 'base') or preview_profile != d['profile']
                label=f"{title} · {'BF16' if preview_profile==BF16 else 'FP32'} · 音色版本 {p['version'][:8]}"+(' · 修改前的试听' if stale else '')
                feeling = '基础情感' if emotion == 'base' else '、'.join(f'{n} {v:g}' for n,v in zip(EMOTION_LABELS,emotion) if v)
                # Plain code fences prevent user text becoming clickable Markdown.
                values += [label+' · '+feeling+'\n\n'+p['text'].replace('<','&lt;'),str(drafts.directory(d['workspaceId'])/p['file'])]
        return values

    def version_status(d):
        p=d.get('currentPackage')
        if not p: return '尚未制作临时音色。'
        if set(p.get('profiles',[])) != {FP32,BF16}:
            return '现有包为单精度，下次生成或下载时补齐 FP32、BF16。'
        if p['referenceKey'] != reference_key(d):
            return '参考已修改，下一次制作、试听或下载时更新音色。'
        if p['key'] != package_key(d):
            return '音色信息已修改，下一次操作会更新名称和性别。'
        return '临时音色已保存 · FP32 + BF16 · 版本 '+p['version'][:8]

    def load(d):
        source=d.get('sourceId')
        if source:
            m=drafts.materials(d['workspaceId']); r=m.record(source); e=d['editor']
            html=player_html(m.directory(source)/'audio.wav',record=r,values=e['rows'],primary=e['primary'],
                peaks=m.waveform(source),analysis=m.vad_analysis(source),workspace_id=d['workspaceId'],editor_state=e)
        else:
            html=player_html()
        f=d['filters']
        vector = d.get('emotion', 'base')
        vector = [0]*8 if vector == 'base' else vector
        return (d['workspaceId'],json.dumps(d,ensure_ascii=False),gr.update(choices=choices(),value=d['workspaceId']),
            gr.update(choices=choices(True),value=None),'已保存 · '+local_time(d['savedAt']),html,
            json.dumps(d['editor'] or {},ensure_ascii=False),d['name'],d['gender'],d['profile'],d['device'],d['text'],
            f['silence'],f['enabled'],gr.update(value=f['volume'],interactive=f['enabled']),f['minimum'],
            gr.update(visible=bool(d['packages'])),*preview_values(d),None,gr.update(visible=False),version_status(d),*vector)
    load_outputs=[wid,context,selector,recycled,save_status,player,editor,name,gender,profile,device,text,
        silence,enabled,volume,minimum,audition_area,current_info,current_audio,download,clear_confirm,result,*emotion_controls]
    def initial(saved,pending):
        try:
            d=drafts.activate(saved) if saved else drafts.current()
        except (OSError,ValueError):
            d=drafts.current()
        if pending:
            try:
                d=drafts.save(d['workspaceId'],pending)
            except Exception as exc:
                values=list(load(d));values[4]='保存失败（刷新前的编辑）：'+str(exc)
                return tuple(values)
        return load(d)
    app.load(initial,[context,payload],load_outputs,js="() => {const id=localStorage.getItem('producer-active-workspace') || ''; return [id,localStorage.getItem(`producer-pending:${id}`) || ''];}",api_name='workspace_load')
    context.change(None,context,None,js=INITIALIZE_WORKFLOW,queue=False)
    ack.change(None,ack,None,js=ACK,queue=False)

    def save(w,encoded):
        if not w or not encoded: return (gr.skip(),)*6
        try:
            d=drafts.save(w,encoded)
            sent=json.loads(encoded)
            return ('已保存 · '+local_time(d['savedAt']),json.dumps({'workspaceId':w,'revision':d['revision'],'clientId':sent['clientId'],'sequence':sent['sequence']}),
                    gr.update(choices=choices()),preview_values(d)[0],None,version_status(d))
        except Exception as exc:
            return ('保存失败：'+str(exc),)+(gr.skip(),)*5
    payload.input(save,[wid,payload],[save_status,ack,selector,current_info,download,result],queue=False,trigger_mode='always_last',api_name='workspace_save')

    def flush(w,encoded):
        if w and encoded: return drafts.save(w,encoded)
        raise ValueError('工作区尚未加载')

    def move(w,target,encoded):
        flush(w,encoded); drafts.cancel(w,wait=True)
        return load(drafts.activate(target))
    selector.input(move,[wid,selector,payload],load_outputs,js=FLUSH_THREE,concurrency_limit=None,api_name='workspace_switch')
    def create(w,encoded):
        flush(w,encoded); drafts.cancel(w,wait=True); return load(drafts.new())
    new.click(create,[wid,payload],load_outputs,js=FLUSH,concurrency_limit=None,api_name='workspace_new')
    def recover(w,encoded):
        flush(w,encoded); drafts.cancel(w,wait=True); return load(drafts.last())
    last.click(recover,[wid,payload],load_outputs,js=FLUSH,concurrency_limit=None,api_name='workspace_last')
    clear.click(lambda:gr.update(visible=True),outputs=clear_confirm,queue=False)
    dismiss.click(lambda:gr.update(visible=False),outputs=clear_confirm,queue=False)
    def empty(w,encoded):
        flush(w,encoded); return load(drafts.clear(w))
    confirm.click(empty,[wid,payload],load_outputs,js=FLUSH,concurrency_limit=None,api_name='workspace_clear')
    def undo_clear(w,target,encoded):
        flush(w,encoded); drafts.cancel(w,wait=True); return load(drafts.restore(target))
    restore.click(undo_clear,[wid,recycled,payload],load_outputs,js=FLUSH_THREE,concurrency_limit=None,api_name='workspace_restore')
    def upload(w,path,encoded,progress=gr.Progress()):
        flush(w,encoded); drafts.cancel(w,wait=True); return load(drafts.import_media(w,path,progress))
    media.upload(upload,[wid,media,payload],load_outputs,js=FLUSH_THREE,concurrency_limit=None,api_name='workspace_import')
    def old(w,source,encoded):
        flush(w,encoded); drafts.cancel(w,wait=True); return load(drafts.import_legacy(w,source))
    import_legacy.click(old,[wid,legacy,payload],load_outputs,js=FLUSH_THREE,concurrency_limit=None)

    def job(kind):
        def run(w,encoded,progress=gr.Progress()):
            try:
                flush(w,encoded)
                entry,d=drafts.run(w,kind,progress)
                message=('试听完成' if kind=='audition' else '最新音色包已准备好 · FP32 + BF16' if kind=='export' else '双精度音色包已生成 · FP32 + BF16')+' · 版本 '+entry['version'][:8]
                if package_key(d) != entry['key']:
                    message += ' · 草稿已继续修改，下次操作会更新。'
                path=drafts.export_path(w,entry) if kind=='export' else None
                sent=json.loads(encoded)
                # Only a successful explicit audition may replace the audio player.
                # Resending the same file during a build/export resets its playback.
                info,audio=preview_values(d)
                if kind != 'audition': audio=gr.skip()
                return message,gr.update(visible=True),info,audio,path,'已保存 · '+local_time(d['savedAt']),json.dumps({'workspaceId':w,'revision':d['revision'],'clientId':sent['clientId'],'sequence':sent['sequence']})
            except Exception as exc:
                return str(exc)+'；成功试听结果仍保留。',gr.skip(),gr.skip(),gr.skip(),None,gr.skip(),gr.skip()
        return run
    outputs=[result,audition_area,current_info,current_audio,download,save_status,ack]
    for button,kind in ((make,'build'),(audition,'audition'),(export,'export')):
        button.click(job(kind),[wid,payload],outputs,js=FLUSH,concurrency_limit=None,api_name='workspace_'+kind)
    cancel.click(lambda w:drafts.cancel(w),wid,result,queue=False,api_name='workspace_cancel')
    list_api=gr.Button(visible=False)
    list_api.click(lambda:gr.update(choices=choices()),outputs=selector,api_name='workspace_list')
    return drafts
