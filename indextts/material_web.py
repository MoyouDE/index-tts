"""Three-step material workflow, independent of producer model internals."""
from types import SimpleNamespace
import copy
import math
import uuid
import gradio as gr
from .web_common import ui_errors
from .material_player import player_html, MARK_POSITION, PREVIEW_RANGE


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


def keep_range(values, active, start, end, duration):
    start, end = float(start), float(end)
    if not math.isfinite(start+end) or not 0 <= start < end <= duration:
        raise ValueError("截取起点必须小于终点，且在素材范围内")
    if end-start > 15+1e-6:
        raise ValueError("每段最多 15 秒，请缩短截取范围；长素材可保留多段")
    updated = copy.deepcopy(values)
    target = next((r for r in updated if r[0] == active), None)
    if active != "__new__" and target is None:
        raise ValueError("待用片段已变更，请重新选择")
    if any(r[3] and r[0] != active and start < float(r[2])-1e-6 and end > float(r[1])+1e-6 for r in updated):
        raise ValueError("截取范围与其他待用片段重叠，请调整边界或先移除重复片段")
    if target is None:
        active = "s" + uuid.uuid4().hex[:10]
        updated.append([active, start, end, True])
    else:
        target[:] = [active, start, end, True]
    return sorted(updated, key=lambda r: float(r[1])), active


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
    gr.Markdown("### 2. 在完整音轨上截取")
    main_player = gr.HTML(player_html(), label="完整音轨", show_label=True, container=True)
    with gr.Row():
        smart = gr.Button("智能截取 · 重新生成建议片段")
        clear = gr.Button("清空待用片段", size="sm")
    preview_id = gr.Dropdown(choices=[("新增截取", "__new__")], value="__new__", label="待用片段（选择后可调整）", interactive=True)
    with gr.Row():
        start = gr.Slider(0, 1, value=0, step=.001, label="截取起点（秒）", interactive=True)
        end = gr.Slider(0, 1, value=1, step=.001, label="截取终点（秒）", interactive=True)
    with gr.Row():
        mark_start = gr.Button("当前位置设为起点", size="sm")
        mark_end = gr.Button("当前位置设为终点", size="sm")
        play = gr.Button("试听截取范围", size="sm")
    with gr.Row():
        keep = gr.Button("保留这段", variant="primary")
        remove = gr.Button("移除当前片段")
    gr.Markdown("拖动起止位置或直接输入秒数，再点击“保留这段”。每段最多 15 秒；长素材可以保留多段。智能截取会替换页面中的片段草稿，已保存选择不变；建议仍需排除其他人声与背景干扰。")
    details = gr.Textbox(label="当前选择", lines=2, interactive=False)
    with gr.Accordion("编辑片段（边界、拆分、合并、主参考）", open=False):
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
        chosen = [r[0] for r in values if r[3]]
        main = main if main in chosen else "auto"
        preview = preview if preview == "__new__" or preview in chosen else (resolve_primary(values, main) if chosen else "__new__")
        return (gr.update(choices=[("新增截取", "__new__")]+labels([r for r in values if r[3]]), value=preview),
                gr.update(choices=[("自动选择", "auto")]+[(r[0], r[0]) for r in values if r[3]], value=main))

    @ui_errors
    def load(source_id):
        if not source_id:
            return [], *view([]), None, "", player_html(), False, gr.update(maximum=1, value=0), gr.update(maximum=1, value=1)
        record = materials.record(source_id)
        values = table_rows(record)
        main = record["primary"] or "auto"
        selected = next((r for r in values if r[0] == main), None)
        duration = record["durationSeconds"]
        return (values, *view(values, main), record["revision"], summary(record, values, main),
                player_html(materials.directory(source_id)/"audio.wav"), False,
                gr.update(maximum=duration, value=selected[1] if selected else 0),
                gr.update(maximum=duration, value=selected[2] if selected else min(15, duration)))

    loaded_outputs = [rows, preview_id, primary, revision, details, main_player, confirmed, start, end]
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
        if not source_id:
            return *view([]), "", False
        return *view(values, main, preview), summary(materials.record(source_id), values, main), False
    sync_outputs = [preview_id, primary, details, confirmed]
    rows.input(sync_rows, [source, rows, primary, preview_id], sync_outputs)

    @ui_errors
    def select_range(source_id, values, active, a, b):
        if not source_id:
            return gr.update(maximum=1, value=0), gr.update(maximum=1, value=1)
        duration = materials.record(source_id)["durationSeconds"]
        row = next((r for r in values if r[0] == active and r[3]), None)
        return (gr.update(maximum=duration, value=row[1] if row else min(float(a), duration)),
                gr.update(maximum=duration, value=row[2] if row else min(float(b), duration)))
    preview_id.change(select_range, [source, rows, preview_id, start, end], [start, end])
    rows.change(select_range, [source, rows, preview_id, start, end], [start, end])

    @ui_errors
    def retain(source_id, values, active, a, b, main):
        if not source_id:
            raise ValueError("请先上传或选择素材")
        record = materials.record(source_id)
        values, active = keep_range(values, active, a, b, record["durationSeconds"])
        return values, *view(values, main, active), summary(record, values, main), False
    keep.click(retain, [source, rows, preview_id, start, end, primary], [rows, *sync_outputs])

    @ui_errors
    def discard(source_id, values, active, main):
        if active == "__new__" or not any(r[0] == active and r[3] for r in values):
            raise ValueError("请先选择一个待用片段")
        values = [[*r[:3], r[3] if r[0] != active else False] for r in values]
        return values, *view(values, main), summary(materials.record(source_id), values, main), False
    remove.click(discard, [source, rows, preview_id, primary], [rows, *sync_outputs])

    @ui_errors
    def clear_ranges(source_id, values):
        if not source_id:
            raise ValueError("请先上传或选择素材")
        values = [[*r[:3], False] for r in values]
        return values, *view(values), summary(materials.record(source_id), values, "auto"), False
    clear.click(clear_ranges, [source, rows], [rows, *sync_outputs])

    @ui_errors
    def suggest(source_id, progress=gr.Progress()):
        if not source_id:
            raise ValueError("请先上传或选择素材")
        record = materials.record(source_id)
        detected = materials.suggest_segments(source_id, progress)
        values = table_rows(detected)
        return values, *view(values), summary({**record, **detected}, values, "auto"), False
    smart.click(suggest, source, [rows, *sync_outputs], concurrency_id="materials", concurrency_limit=1)
    mark_start.click(None, start, start, js=MARK_POSITION, queue=False)
    mark_end.click(None, end, end, js=MARK_POSITION, queue=False)
    play.click(None, [start, end], js=PREVIEW_RANGE, queue=False)
    start.input(lambda: False, outputs=confirmed)
    end.input(lambda: False, outputs=confirmed)
    split.click(ui_errors(edit_split), [rows, preview_id, point], rows).success(sync_rows, [source, rows, primary, preview_id], sync_outputs)
    merge.click(ui_errors(edit_merge), [rows, preview_id], rows).success(sync_rows, [source, rows, primary, preview_id], sync_outputs)
    primary.input(sync_rows, [source, rows, primary, preview_id], sync_outputs)

    @ui_errors
    def prepare(source_id, values, main, rev, approved, construction, active, a, b, *args):
        if not source_id:
            raise ValueError("请先上传或选择素材")
        row = next((r for r in values if r[0] == active and r[3]), None)
        if row is None or abs(float(a)-float(row[1])) > 1e-6 or abs(float(b)-float(row[2])) > 1e-6:
            raise ValueError("截取范围尚未保留，请先点击“保留这段”")
        display_name = args[0]
        if not display_name or not display_name.strip() or len(display_name.strip()) > 128:
            raise ValueError("请填写音色名称（1～128 个字符）")
        selection, record = prepare_selection(materials, source_id, values, main, rev, approved, construction)
        return {"selection": selection, "args": copy.deepcopy(args)}, record["revision"], summary(record, values, selection["primary"])
    # Save before encoding so a model failure still leaves the session revision current.
    prepared = make.click(prepare, [source, rows, primary, revision, confirmed, method, preview_id, start, end, *inputs],
                          [snapshot, revision, details], concurrency_id="materials", concurrency_limit=1)
    def generate_snapshot(job, progress=gr.Progress()):
        return generate(copy.deepcopy(job["selection"]), *copy.deepcopy(job["args"]), progress=progress)
    built = prepared.success(generate_snapshot, snapshot, outputs, concurrency_limit=None)
    return SimpleNamespace(source=source, choices=choices, built=built)
