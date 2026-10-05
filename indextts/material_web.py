"""Three-step material workflow, independent of producer model internals."""
from types import SimpleNamespace
import copy
import math
import uuid
import json
import gradio as gr
from .web_common import ui_errors
from .material_player import player_html, INITIALIZE


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


def exclude_short_ranges(values, minimum_seconds):
    if (isinstance(minimum_seconds, bool) or not isinstance(minimum_seconds, (int, float))
            or not math.isfinite(minimum_seconds) or not 0 < minimum_seconds <= 15):
        raise ValueError("最短保留时长须大于 0 且不超过 15 秒")
    updated = copy.deepcopy(values)
    excluded = 0
    for row in updated:
        duration = float(row[2]) - float(row[1])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("片段起止时间无效，请先调整边界")
        if row[3] and duration < minimum_seconds - 1e-6:
            row[3] = False
            excluded += 1
    return updated, excluded


def segmentation_options(silence_seconds, volume_enabled, volume_db):
    if (isinstance(silence_seconds, bool) or not isinstance(silence_seconds, (int, float))
            or not math.isfinite(silence_seconds) or not .1 <= silence_seconds <= 3):
        raise ValueError("空白间隔须为 0.1～3 秒")
    if not isinstance(volume_enabled, bool):
        raise ValueError("最低音量过滤开关无效")
    return {"min_silence_ms": round(silence_seconds * 1000, 6),
            "min_volume_db": volume_db if volume_enabled else None}


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


def prepare_selection(materials, source_id, values, main, revision):
    primary = resolve_primary(values, main)
    chosen = [r for r in values if r[3]]
    if any(float(r[2])-float(r[1]) > 15+1e-6 for r in chosen):
        raise ValueError("每个选中片段不得超过 15 秒，请在“编辑片段”中拆分")
    construction = "speaker-mean-v1" if len(chosen) > 1 else "primary-only-v1"
    record = materials.save(source_id, values, primary, expected_revision=revision)
    selection = materials.selection(source_id, record["revision"], confirmed=True)
    selection["method"] = construction
    return selection, record


def read_draft(source_id, draft):
    try:
        value = json.loads(draft)
        if value["sourceId"] != source_id or not isinstance(value["rows"], list):
            raise ValueError()
        rows, main = value["rows"], value["primary"]
        if not isinstance(main, str) or any(len(r) != 4 or not isinstance(r[0], str)
                or not isinstance(r[3], bool) or isinstance(r[1], bool) or isinstance(r[2], bool)
                or not isinstance(r[1], (float, int)) or not isinstance(r[2], (float, int))
                or not math.isfinite(r[1]+r[2]) for r in rows):
            raise ValueError()
        return copy.deepcopy(rows), main
    except (ValueError, KeyError, TypeError):
        raise ValueError("音轨选择未就绪或与当前素材不一致，请重新选择素材") from None


def build_controls(materials, session, *, generate, session_directory, inputs, outputs,
                   name, options, diagnostics, show_tools=False):
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

    gr.Markdown("### ① 上传音频")
    media = gr.File(label="拖入音频或视频，自动寻找片段", type="filepath", file_types=["audio", "video"], height=110)
    with gr.Accordion("使用已保存素材", open=False):
        with gr.Row():
            source = gr.Dropdown(choices=choices(), value=None, label="已保存素材", interactive=True)
            refresh = gr.Button("刷新素材", size="sm")
    with gr.Row():
        with gr.Column(min_width=190):
            silence = gr.Slider(minimum=.1, maximum=3, value=.5, step=.05, label="分段停顿（秒）",
                            elem_id='material-silence',
                            info="调大可保留更连续的对白。")
        with gr.Column(min_width=190):
            volume_enabled = gr.Checkbox(label="过滤小声杂音", value=False, elem_id='material-volume-enabled')
            volume_floor = gr.Slider(minimum=-80, maximum=0, value=-50, step=1,
                                     label="最低音量（dBFS）", interactive=False, elem_id='material-volume',
                                     info="调高可排除更弱的声音。")
        with gr.Column(min_width=190):
            minimum_duration = gr.Slider(minimum=.1, maximum=15, value=1, step=.1,
                                     label="最短片段（秒）", elem_id='material-minimum', interactive=True,
                                     info="调大可隐藏零碎短声音。")
    volume_enabled.input(lambda enabled: gr.update(interactive=enabled), volume_enabled, volume_floor)
    main_player = gr.HTML(player_html(), label="候选与片段调整", show_label=False, container=True)
    draft = gr.Textbox(value="", elem_id="material-waveform-draft", show_label=False)
    main_player.change(None, js=INITIALIZE, queue=False)
    details = gr.Textbox(label="当前选择", lines=2, interactive=False, visible=False)
    preview_id = gr.Dropdown(choices=[("新增截取", "__new__")], value="__new__", visible=False)
    primary = gr.Dropdown(choices=[("自动选择", "auto")], value="auto", visible=False)
    rows = gr.Dataframe(headers=["片段 ID", "起点（秒）", "终点（秒）", "选中"],
                            datatype=["str", "number", "number", "bool"], type="array", interactive=False,
                            col_count=(4, "fixed"), visible=False)
    gr.Markdown("### ④ 使用选中片段生成音色包")
    name.render()
    with gr.Accordion("高级设置", open=False):
        for component in options:
            component.render()
        gr.Markdown("按选中片段数量自动处理：一段使用单段参考，多段等权融合音色身份；声学提示仍使用主参考。不会自动清除背景音乐。")
    gr.Markdown("生成前建议逐段试听：选中片段应来自同一个目标人物，尽量避开其他人声、杂音和背景音乐。",
                elem_id="material-quality-hint")
    make = gr.Button("生成音色包", variant="primary")
    outputs[2].render()
    outputs[0].render()
    with gr.Accordion("诊断与模型", open=False, visible=show_tools):
        diagnostics()
    revision = gr.State(None)
    snapshot = gr.State(None)
    analysis = gr.State(None)

    def view(values, main="auto", preview=None):
        chosen = [r[0] for r in values if r[3]]
        main = main if main in chosen else "auto"
        preview = preview if preview == "__new__" or preview in chosen else (resolve_primary(values, main) if chosen else "__new__")
        return (gr.update(choices=[("新增截取", "__new__")]+labels([r for r in values if r[3]]), value=preview),
                gr.update(choices=[("自动选择", "auto")]+[(r[0], r[0]) for r in values if r[3]], value=main))

    def current_record(source_id, detected=None):
        record = materials.record(source_id)
        if detected and detected.get("sourceId") == source_id:
            record["speechSeconds"] = detected["speechSeconds"]
        return record

    def editor(source_id, values, main, detected=None):
        record = current_record(source_id, detected)
        return player_html(materials.directory(source_id)/"audio.wav", record=record, values=values,
                           primary=main, peaks=materials.waveform(source_id), analysis=materials.vad_analysis(source_id))

    def payload(source_id, values, main):
        return json.dumps({"sourceId": source_id, "rows": values, "primary": main}, ensure_ascii=False)

    @ui_errors
    def load(source_id):
        if not source_id:
            return ([], *view([]), None, "", player_html(), "", None, .5, False,
                    gr.update(value=-50, interactive=False))
        record = materials.record(source_id)
        values = table_rows(record)
        main = record["primary"] or "auto"
        settings = record.get("settings", {})
        minimum_volume = settings.get("minVolumeDb")
        detected = {"sourceId": source_id, "speechSeconds": record["speechSeconds"]}
        return (values, *view(values, main), record["revision"], summary(record, values, main),
                editor(source_id, values, main), payload(source_id, values, main), detected,
                settings.get("minSilenceMs", 500) / 1000, minimum_volume is not None,
                gr.update(value=minimum_volume if minimum_volume is not None else -50,
                          interactive=minimum_volume is not None))

    loaded_outputs = [rows, preview_id, primary, revision, details, main_player, draft,
                      analysis, silence, volume_enabled, volume_floor]
    source.change(load, source, loaded_outputs)
    refresh.click(lambda: gr.update(choices=choices(), value=None), outputs=source)

    @ui_errors
    def import_material(path, silence_seconds, volume_enabled, volume_db, progress=gr.Progress()):
        if not path:
            return gr.update()
        record = materials.import_media(path, progress,
            **segmentation_options(silence_seconds, volume_enabled, volume_db))
        return gr.update(choices=choices(), value=record["sourceId"])
    media.upload(import_material, [media, silence, volume_enabled, volume_floor], source,
                 concurrency_id="materials", concurrency_limit=1)
    media.clear(lambda: gr.update(value=None), outputs=source)

    @ui_errors
    def sync_draft(source_id, encoded):
        if not source_id or not encoded:
            return gr.skip(), gr.skip(), gr.skip()
        values, main = read_draft(source_id, encoded)
        return values, *view(values, main)
    draft.input(ui_errors(sync_draft), [source, draft], [rows, preview_id, primary],
                trigger_mode="always_last", show_progress="hidden")


    @ui_errors
    def prepare(source_id, values, main, rev, encoded, *args):
        if not source_id:
            raise ValueError("请先上传或选择素材")
        # The latest browser draft is authoritative, even if table mirroring is still queued.
        values, main = read_draft(source_id, encoded)
        display_name = args[0]
        if not display_name or not display_name.strip() or len(display_name.strip()) > 128:
            raise ValueError("请填写音色名称（1～128 个字符）")
        selection, record = prepare_selection(materials, source_id, values, main, rev)
        return {"selection": selection, "args": copy.deepcopy(args)}, record["revision"], summary(record, values, selection["primary"])
    # Save before encoding so a model failure still leaves the session revision current.
    prepared = make.click(prepare, [source, rows, primary, revision, draft, *inputs],
                          [snapshot, revision, details], concurrency_id="materials", concurrency_limit=1)
    def generate_snapshot(job, progress=gr.Progress()):
        return generate(copy.deepcopy(job["selection"]), *copy.deepcopy(job["args"]), progress=progress)
    built = prepared.success(generate_snapshot, snapshot, outputs, concurrency_limit=None)
    return SimpleNamespace(source=source, choices=choices, built=built)
