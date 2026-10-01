"""Producer page. It has no emotion or audition page dependency."""
from pathlib import Path
from types import SimpleNamespace
import gradio as gr
from .validation_common import audio_details
from .library_web import library_controls
from .web_common import choices


def build_page(service, library, session, *, source_model_dir):
    source_model_dir = str(Path(source_model_dir).expanduser().resolve())

    def preview(path):
        if not path:
            return None, {}
        try:
            return path, audio_details(path)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def build(reference, vid, name, gender, profile, device, session, progress=gr.Progress()):
        try:
            return service.build(reference, vid, name, gender, profile, device, source_model_dir, session, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def inspect(pack, session):
        try:
            return service.inspect(pack, source_model_dir, session)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def build_from_selection(selection, vid, name, gender, profile, device, sid, progress=None):
        return service.build_selection(selection,vid,name,gender,profile,device,source_model_dir,sid,progress)

    with gr.Tab("音色包生成", id="voices"):
        gr.Textbox(label="源模型目录", value=source_model_dir, interactive=False,
                   info="制包使用的源模型权重，固定于工具启动配置；音色包保存在下方本地工作区。")
        gr.Markdown("短参考音频可直接制包；视频或长录音请使用下方“长视频／录音素材整理”，选择并检查主参考。")
        with gr.Row():
            with gr.Column():
                reference = gr.File(label="参考音频", file_types=["audio"], type="filepath")
                audio = gr.Audio(label="参考音频播放", interactive=False)
                info = gr.JSON(label="音频信息（超过 15 秒时仅使用前 15 秒）")
            with gr.Column():
                vid = gr.Textbox(label="音色 ID", placeholder="例如 reader-voice-01")
                name = gr.Textbox(label="音色名称")
                gender = gr.Dropdown([("未知", "unknown"), ("女声", "female"), ("男声", "male"), ("中性", "neutral")], value="unknown", label="性别")
                profile = gr.Radio([("FP32", "compatible-fp32"), ("BF16 混合精度", "fixed-voice-bf16")], value="compatible-fp32", label="制包精度")
                device = gr.Dropdown([("自动", "auto"), ("CPU", "cpu"), ("CUDA 0", "cuda:0")], value="auto", allow_custom_value=True, label="计算设备")
                gr.Markdown("BF16 音色包需要匹配的 BF16 推理模型。设备能执行 BF16 不代表一定更快。")
                make = gr.Button("生成并校验音色包", variant="primary")
                unload = gr.Button("卸载制包模型")
                status = gr.Textbox(label="模型状态", interactive=False)
        output = gr.File(label="下载音色包", interactive=False)
        with gr.Accordion("生成与校验结果", open=False):
            report = gr.JSON(label="制包详情")
        from .material_web import build_controls
        material_page=build_controls(service.materials,session,generate=build_from_selection,
            session_directory=service.session_dir,inputs=[vid,name,gender,profile,device,session],outputs=[output,report])
        reference.change(preview, reference, [audio, info])
        built = make.click(build, [reference, vid, name, gender, profile, device, session], [output, report], concurrency_limit=None)
        unload.click(service.unload_producer, outputs=status, concurrency_id="producer", concurrency_limit=1)
        library_voice, library_profile = library_controls(library, service.rebuild, session, source_model_dir, device)
        built.success(lambda v: gr.update(choices=choices(library), value=v), vid, library_voice)
        material_page.built.success(lambda v: gr.update(choices=choices(library),value=v),vid,library_voice)
        with gr.Accordion("检查已有音色包", open=False):
            pack = gr.File(label="上传 .ivp", file_types=[".ivp"], type="filepath")
            gr.Markdown("检查包结构、完整性以及与工具固定源模型的兼容性。")
            check = gr.Button("检查音色包")
            inspected = gr.JSON(label="包检查结果")
            check.click(inspect, [pack, session], inspected)
    return SimpleNamespace(voice=library_voice, profile=library_profile,materials=material_page)
