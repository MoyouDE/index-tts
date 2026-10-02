"""Three-step material workflow, independent of producer model internals."""
from pathlib import Path
from types import SimpleNamespace
import copy
import uuid
import gradio as gr
from .web_common import ui_errors


def table_rows(record):
    return [[s["id"], s["start"], s["end"], s["selected"]] for s in record["segments"]]


def edit_split(rows, selected, point):
    rows = [list(r) for r in rows]
    row = next((r for r in rows if r[0] == selected), None)
    if row is None or not float(row[1]) < float(point) < float(row[2]):
        raise ValueError("拆分点必须在当前片段内部")
    index = rows.index(row)
    rows[index:index+1] = [[row[0], row[1], float(point), row[3]],
                          ["s"+uuid.uuid4().hex[:10], float(point), row[2], row[3]]]
    return rows


def edit_merge(rows, selected):
    rows = sorted([list(r) for r in rows], key=lambda r: float(r[1]))
    index = next((i for i, r in enumerate(rows) if r[0] == selected), -1)
    if index < 0 or index+1 >= len(rows):
        raise ValueError("请选择有后续片段的片段")
    a, b = rows[index:index+2]
    rows[index:index+2] = [[a[0], min(a[1], b[1]), max(a[2], b[2]), bool(a[3] or b[3])]]
    return rows


def resolve_primary(rows, preferred="auto"):
    chosen = [r for r in rows if r[3]]
    if not chosen:
        raise ValueError("请至少选择一个片段")
    if preferred != "auto" and any(r[0] == preferred for r in chosen):
        return preferred
    # Duration is a deterministic starting recommendation, not a quality score.
    return max(chosen, key=lambda r: float(r[2])-float(r[1]))[0]


def prepare_selection(materials, source_id, values, main, revision, approved, construction):
    if not approved:
        raise ValueError("请确认选中片段属于同一目标人物")
    primary = resolve_primary(values, main)
    chosen = [r for r in values if r[3]]
    if any(float(r[2])-float(r[1]) > 15+1e-6 for r in chosen):
        raise ValueError("每个选中片段不得超过 15 秒，请在“编辑片段”中拆分")
    if construction == "auto":
        construction = "speaker-mean-v1" if len(chosen) > 1 else "primary-only-v1"
    if construction not in {"speaker-mean-v1", "primary-only-v1"}:
        raise ValueError("未知参考构建方法")
    if construction == "speaker-mean-v1" and len(chosen) < 2:
        raise ValueError("多段融合至少需要两个片段；可切换为自动或单段")
    record = materials.save(source_id, values, primary, expected_revision=revision)
    selection = materials.selection(source_id, record["revision"], confirmed=True)
    selection["method"] = construction
    return selection, record


def build_controls(materials, session, *, generate, session_directory, inputs, outputs,
                   name, options, method, diagnostics):
    def choices():
        return [(f"{r['name']} · {r['sourceId'][:8]}", r["sourceId"]) for r in materials.items()]

    def labels(values):
        return [(f"{r[0]} · {float(r[1]):.2f}–{float(r[2]):.2f} 秒（{float(r[2])-float(r[1]):.2f} 秒）", r[0]) for r in values]

    def summary(record, values, main):
        chosen = [r for r in values if r[3]]
        seconds = sum(float(r[2])-float(r[1]) for r in chosen)
        primary = resolve_primary(values, main) if chosen else "未选择"
        return (f"素材 {record['durationSeconds']:.2f} 秒 · 检测语音 {record['speechSeconds']:.2f} 秒\n"
                f"选中 {len(chosen)} 段 / {seconds:.2f} 秒 · 主参考 {primary}")

    gr.Markdown("### 1. 上传素材")
    media = gr.File(label="上传音频或视频 · 自动分段", type="filepath", file_types=["audio", "video"], height=110)
    with gr.Accordion("使用已保存素材", open=False):
        with gr.Row():
            source = gr.Dropdown(choices=choices(), value=None, label="已保存素材", interactive=True)
            refresh = gr.Button("刷新素材", size="sm")
    gr.Markdown("### 2. 听一听，选择要用的片段")
    selected = gr.CheckboxGroup(label="参与制包的片段", choices=[], value=[], interactive=True, elem_id="producer-segments")
    preview_id = gr.Dropdown(label="试听片段", interactive=True)
    segment_audio = gr.Audio(label="片段回放", interactive=False)
    details = gr.Textbox(label="当前选择", lines=2, interactive=False)
    with gr.Accordion("编辑片段（边界、拆分、合并、主参考）", open=False):
        original_audio = gr.Audio(label="完整音轨", interactive=False)
        rows = gr.Dataframe(headers=["片段 ID", "起点（秒）", "终点（秒）", "选中"],
                            datatype=["str", "number", "number", "bool"], type="array", interactive=True,
                            col_count=(4, "fixed"), label="分段编辑 · 生成时自动保存")
        primary = gr.Dropdown(choices=[("自动选择", "auto")], value="auto", label="主参考", interactive=True)
        gr.Markdown("自动主参考选用最长的选中片段；可手动指定。每段最多 15 秒，超长片段请拆分。")
        with gr.Row():
            point = gr.Number(label="拆分位置（素材绝对秒数）")
            split = gr.Button("拆分试听片段")
            merge = gr.Button("与下一片段合并")
    gr.Markdown("### 3. 生成音色包")
    name.render()
    confirmed = gr.Checkbox(label="选中片段都是同一人，且已排除不需要的声音", value=False)
    with gr.Accordion("高级设置", open=False):
        for component in options:
            component.render()
        gr.Markdown("自动模式：单段使用主参考，多段融合身份表示；声学提示仍使用主参考。不会自动清除背景音乐。")
    make = gr.Button("生成音色包", variant="primary")
    outputs[2].render()
    outputs[0].render()
    with gr.Accordion("诊断与模型", open=False):
        diagnostics()
    revision = gr.State(None)
    snapshot = gr.State(None)

    def view(values, main="auto", preview=None):
        ids = [r[0] for r in values]
        chosen = [r[0] for r in values if r[3]]
        main = main if main in chosen else "auto"
        preview = preview if preview in ids else (resolve_primary(values, main) if chosen else (ids[0] if ids else None))
        return (gr.update(choices=labels(values), value=chosen),
                gr.update(choices=labels(values), value=preview),
                gr.update(choices=[("自动选择", "auto")]+[(r[0], r[0]) for r in values if r[3]], value=main))

    @ui_errors
    def load(source_id):
        if not source_id:
            return [], *view([]), None, "", None, False, None
        record = materials.record(source_id)
        values = table_rows(record)
        main = record["primary"] or "auto"
        return (values, *view(values, main), record["revision"], summary(record, values, main),
                str(materials.directory(source_id)/"audio.wav"), False, None)

    loaded_outputs = [rows, selected, preview_id, primary, revision, details, original_audio, confirmed, segment_audio]
    source.change(load, source, loaded_outputs)
    refresh.click(lambda: gr.update(choices=choices(), value=None), outputs=source)

    @ui_errors
    def import_material(path, progress=gr.Progress()):
        if not path:
            return gr.update()
        record = materials.import_media(path, progress)
        return gr.update(choices=choices(), value=record["sourceId"])
    media.upload(import_material, media, source, concurrency_id="materials", concurrency_limit=1)
    media.clear(lambda: gr.update(value=None), outputs=source)

    @ui_errors
    def sync_rows(source_id, values, main, preview):
        return *view(values, main, preview), summary(materials.record(source_id), values, main), False
    sync_outputs = [selected, preview_id, primary, details, confirmed]
    rows.input(sync_rows, [source, rows, primary, preview_id], sync_outputs)

    @ui_errors
    def choose(source_id, values, chosen, main, preview):
        values = [[*r[:3], r[0] in chosen] for r in values]
        _, preview_update, main_update = view(values, main, preview)
        return values, preview_update, main_update, summary(materials.record(source_id), values, main), False
    selected.input(choose, [source, rows, selected, primary, preview_id], [rows, preview_id, primary, details, confirmed])
    split.click(ui_errors(edit_split), [rows, preview_id, point], rows).success(sync_rows, [source, rows, primary, preview_id], sync_outputs)
    merge.click(ui_errors(edit_merge), [rows, preview_id], rows).success(sync_rows, [source, rows, primary, preview_id], sync_outputs)
    primary.input(sync_rows, [source, rows, primary, preview_id], sync_outputs)

    @ui_errors
    def play_segment(source_id, values, sid, session_id):
        if not source_id or not sid:
            return None
        row = next((r for r in values if r[0] == sid), None)
        if row is None:
            raise ValueError("请选择片段")
        destination = Path(session_directory(session_id))/(uuid.uuid4().hex+".wav")
        return materials.crop(source_id, float(row[1]), float(row[2]), destination)
    # Boundary edits refresh the preview even when the selected ID did not change.
    preview_id.change(play_segment, [source, rows, preview_id, session], segment_audio)
    rows.change(play_segment, [source, rows, preview_id, session], segment_audio)

    @ui_errors
    def prepare(source_id, values, main, rev, approved, construction, *args):
        display_name = args[0]
        if not display_name or not display_name.strip() or len(display_name.strip()) > 128:
            raise ValueError("请填写音色名称（1～128 个字符）")
        selection, record = prepare_selection(materials, source_id, values, main, rev, approved, construction)
        return {"selection": selection, "args": copy.deepcopy(args)}, record["revision"], summary(record, values, selection["primary"])
    # Save before encoding so a model failure still leaves the session revision current.
    prepared = make.click(prepare, [source, rows, primary, revision, confirmed, method, *inputs],
                          [snapshot, revision, details], concurrency_id="materials", concurrency_limit=1)
    def generate_snapshot(job, progress=gr.Progress()):
        return generate(copy.deepcopy(job["selection"]), *copy.deepcopy(job["args"]), progress=progress)
    built = prepared.success(generate_snapshot, snapshot, outputs, concurrency_limit=None)
    return SimpleNamespace(source=source, choices=choices, built=built)
