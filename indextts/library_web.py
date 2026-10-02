"""Local-library controls; production is supplied as a callback."""
from pathlib import Path
import gradio as gr
from .runtime.profiles import FP32, BF16
from .web_common import choices, ui_errors

PROFILES = [("FP32", FP32), ("BF16", BF16)]

def library_controls(library, regenerate_voice, session, source_model_dir, device):
    gr.Markdown("### 本地音色库\n生成后自动保存，选择音色可回放参考、下载或管理。")
    selected = gr.Dropdown(choices(library), value=None, label="库中音色", interactive=True)
    refresh = gr.Button("刷新音色库", size="sm")
    with gr.Column(visible=False) as selected_details:
        version_status = gr.Textbox(label="版本状态", interactive=False)
        reference = gr.Audio(label="库存参考音频", interactive=False)
        package = gr.File(label="下载所选版本", interactive=False)
        with gr.Accordion("管理所选音色", open=False):
            profile = gr.Radio(PROFILES, value=FP32, label="库中版本")
            rebuild = gr.Button("生成／替换所选精度")
            notes = gr.Textbox(label="本地备注", lines=2)
            save = gr.Button("保存备注")
            with gr.Accordion("音色包元数据", open=False):
                details = gr.JSON(label="音色信息与版本状态")
                gr.Textbox(value=str(library.root), label="本地工作区（不进入 Git）", interactive=False)
            with gr.Accordion("删除本地音色", open=False):
                deletion = gr.Textbox(label="将删除的音色及范围", interactive=False)
                confirm = gr.Checkbox(label="确认删除此音色的参考音频、两档包及试听结果", value=False)
                delete = gr.Button("删除选中音色", variant="stop")
                token = gr.State(None)
    status = gr.Textbox(label="音色库操作结果", interactive=False)
    with gr.Accordion("导入已有音色包", open=False):
        imported = gr.File(label="导入 .ivp（无参考音频）", file_types=[".ivp"])
        import_button = gr.Button("导入本地库")

    @ui_errors
    def show(voice, precision):
        if not voice:
            return {}, None, None, "", "", None, False, "", gr.update(visible=False)
        record = library.record(voice)
        item = next(item for item in library.items() if item["voiceId"] == voice)
        audio = None
        if record["reference"]:
            audio = (library.directory(voice) / record["reference"]).resolve()
            if audio.parent != library.directory(voice):
                raise ValueError("参考音频路径无效")
        pack = library.pack_path(voice, precision)
        states = " · ".join(f"{label}: {'已生成' if item['variants'][p].get('ready') else item['variants'][p].get('error', '尚未生成')}" for label, p in PROFILES)
        return (item, str(audio) if audio else None, str(pack) if pack.exists() else None, record["notes"],
                f"{record['displayName']}（{voice}）：参考音频、所有精度包、试听及备注", f"{voice}:{record['instance']}", False, states, gr.update(visible=True))

    outputs = [details, reference, package, notes, deletion, token, confirm, version_status, selected_details]
    selected.change(show, [selected, profile], outputs)
    profile.change(show, [selected, profile], outputs)
    refresh.click(lambda: gr.update(choices=choices(library), value=None), outputs=selected)
    save.click(ui_errors(lambda voice, note: (library.notes(voice, note), "备注已保存")[1]), [selected, notes], status)

    @ui_errors
    def regenerate(voice, precision, dev, sid, progress=gr.Progress()):
        if not voice:
            raise ValueError("请先选择库中音色")
        path, result = regenerate_voice(voice, precision, dev, source_model_dir, sid, progress)
        return path, f"{precision} 生成并校验完成：{result['seconds']:.2f} 秒"

    rebuild.click(regenerate, [selected, profile, device, session], [package, status], concurrency_limit=None).then(show, [selected, profile], outputs)

    @ui_errors
    def import_pack(path):
        if not path:
            raise ValueError("请上传 .ivp")
        installed = library.install(path)
        return gr.update(choices=choices(library), value=Path(installed).parent.name), "音色包已导入"

    import_button.click(import_pack, imported, [selected, status])

    @ui_errors
    def remove(voice, target, approved):
        if not approved:
            raise ValueError("请勾选确认，并核对将删除的音色")
        library.delete(voice, target)
        return gr.update(choices=choices(library), value=None), "音色及本地产物已删除"

    delete.click(remove, [selected, token, confirm], [selected, status])
    return selected, profile
