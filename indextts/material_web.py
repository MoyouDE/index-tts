"""Material preparation controls, independent of producer model internals."""
from pathlib import Path
from types import SimpleNamespace
import uuid
import gradio as gr
from .web_common import ui_errors


def table_rows(record):
    return [[s["id"],s["start"],s["end"],s["selected"]] for s in record["segments"]]


def edit_split(rows, selected, point):
    rows = [list(r) for r in rows]
    row = next((r for r in rows if r[0]==selected),None)
    if row is None or not float(row[1]) < float(point) < float(row[2]):
        raise ValueError("拆分点必须在当前片段内部")
    index = rows.index(row)
    rows[index:index+1] = [[row[0],row[1],float(point),row[3]],
                          ["s"+uuid.uuid4().hex[:10],float(point),row[2],row[3]]]
    return rows


def edit_merge(rows, selected):
    rows = sorted([list(r) for r in rows],key=lambda r: float(r[1]))
    index = next((i for i,r in enumerate(rows) if r[0]==selected),-1)
    if index < 0 or index+1 >= len(rows):
        raise ValueError("请选择有后续片段的片段")
    a,b = rows[index:index+2]
    rows[index:index+2] = [[a[0],min(a[1],b[1]),max(a[2],b[2]),bool(a[3] or b[3])]]
    return rows


def build_controls(materials, session, *, generate, session_directory, inputs, outputs):
    def choices():
        return [(f"{r['name']} · {r['sourceId'][:8]}",r["sourceId"]) for r in materials.items()]

    def summary(record):
        chosen = [s for s in record["segments"] if s["selected"]]
        seconds=sum(s["end"]-s["start"] for s in chosen)
        overlong=[s["id"] for s in chosen if s["end"]-s["start"]>15+1e-6]
        text=(f"素材 {record['durationSeconds']:.2f} 秒 · 检测语音 {record['speechSeconds']:.2f} 秒 · "
              f"选中 {seconds:.2f} 秒（{len(chosen)} 段）\n"
              f"主参考：{record['primary'] or '尚未指定'}；单段模式仅使用主参考，融合模式使用所有选中片段。")
        return text+("\n请拆分超长片段："+", ".join(overlong) if overlong else "")

    with gr.Accordion("长视频／录音素材整理",open=True):
        gr.Markdown("导入完整素材并选择目标人物片段。单段模式使用主参考；可选身份等权融合使用所有选中片段，声学提示与基础情感仍固定为主参考；不会默认取长素材开头。")
        media = gr.File(label="长素材（音频或视频）",type="filepath")
        import_button = gr.Button("导入并自动分段")
        with gr.Row():
            source = gr.Dropdown(choices=choices(),value=None,label="已保存素材",interactive=True)
            refresh = gr.Button("刷新素材列表")
        original_audio = gr.Audio(label="完整音轨回放",interactive=False)
        details = gr.Textbox(label="素材与选择统计",lines=3,interactive=False,
                             info="检测语音区间包含正常短停顿；每个片段最长 15 秒；融合不自动删除内部空白。")
        rows = gr.Dataframe(headers=["片段 ID","起点（秒）","终点（秒）","选中"],
                            datatype=["str","number","number","bool"],type="array",interactive=True,
                            col_count=(4,"fixed"),label="分段编辑（调整后请保存；每个选中片段最多 15 秒）")
        with gr.Row():
            preview_id = gr.Dropdown(label="预览／操作片段",interactive=True)
            primary = gr.Dropdown(label="主参考片段",interactive=True)
        segment_audio = gr.Audio(label="片段回放",interactive=False)
        with gr.Row():
            play = gr.Button("播放当前边界")
            point = gr.Number(label="拆分位置（素材绝对秒数）")
            split = gr.Button("在指定位置拆分")
            merge = gr.Button("与下一片段合并")
        save = gr.Button("保存分段与选择")
        confirmed = gr.Checkbox(label="我已确认选中片段属于同一目标人物，并排除不需要的声音",value=False)
        method = gr.Radio([("单段主参考", "primary-only-v1"), ("多段身份等权融合（待成品验收）", "speaker-mean-v1")], value="primary-only-v1", label="参考条件构建方法")
        make = gr.Button("按所选方法生成音色包",variant="primary")
        status = gr.Textbox(label="素材操作结果",interactive=False)
        revision = gr.State(None)

    @ui_errors
    def load(source_id):
        if not source_id:
            return [],gr.update(choices=[],value=None),gr.update(choices=[],value=None),None,"",None,False
        record=materials.record(source_id)
        ids=[s["id"] for s in record["segments"]]
        preview=record["primary"] or (ids[0] if ids else None)
        return table_rows(record),gr.update(choices=ids,value=preview),gr.update(choices=ids,value=record["primary"]),record["revision"],summary(record),str(materials.directory(source_id)/"audio.wav"),False

    loaded_outputs=[rows,preview_id,primary,revision,details,original_audio,confirmed]
    source.change(load,source,loaded_outputs)
    refresh.click(lambda:gr.update(choices=choices(),value=None),outputs=source)

    @ui_errors
    def import_material(path, progress=gr.Progress()):
        record=materials.import_media(path,progress)
        return gr.update(choices=choices(),value=record["sourceId"]),"导入完成；请试听并确认目标人物，修改分段后保存"

    import_button.click(import_material,media,[source,status],concurrency_id="materials",concurrency_limit=1)

    def update_choices(values):
        ids=[r[0] for r in values if r[0]]
        return gr.update(choices=ids),gr.update(choices=ids),False
    rows.input(update_choices,rows,[preview_id,primary,confirmed])
    split.click(ui_errors(edit_split),[rows,preview_id,point],rows).then(update_choices,rows,[preview_id,primary,confirmed])
    merge.click(ui_errors(edit_merge),[rows,preview_id],rows).then(update_choices,rows,[preview_id,primary,confirmed])
    primary.input(lambda:False,outputs=confirmed)

    @ui_errors
    def play_segment(source_id, values, sid, session_id):
        row=next((r for r in values if r[0]==sid),None)
        if row is None:
            raise ValueError("请选择片段")
        destination=Path(session_directory(session_id))/(uuid.uuid4().hex+".wav")
        return materials.crop(source_id,float(row[1]),float(row[2]),destination)
    play.click(play_segment,[source,rows,preview_id,session],segment_audio)

    @ui_errors
    def save_selection(source_id, values, main, rev):
        record=materials.save(source_id,values,main,expected_revision=rev)
        return record["revision"],summary(record),"选择已保存；修改已制包的选择后，请使用新的音色 ID",False
    save.click(save_selection,[source,rows,primary,revision],[revision,details,status,confirmed])

    @ui_errors
    def build_material(source_id, values, main, rev, approved, construction, *args, progress=gr.Progress()):
        record=materials.record(source_id)
        if values != table_rows(record) or main != record["primary"]:
            raise ValueError("分段或主参考尚未保存，请先保存修改")
        selection=materials.selection(source_id,rev,confirmed=approved)
        selection["method"] = construction
        return generate(selection,*args,progress=progress)
    built=make.click(build_material,[source,rows,primary,revision,confirmed,method,*inputs],outputs,concurrency_limit=None)
    return SimpleNamespace(source=source,choices=choices,built=built)
