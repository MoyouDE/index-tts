"""Producer page. It has no emotion or audition page dependency."""
from types import SimpleNamespace
import gradio as gr
from .validation_common import audio_details
from .library_web import library_controls
from .web_common import choices


def build_page(service, library, session, *, source_model_dir):
    def preview(path):
        if not path:
            return None, {}
        try:
            return path, audio_details(path)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def build(reference, vid, name, gender, profile, device, source, session, progress=gr.Progress()):
        try:
            return service.build(reference, vid, name, gender, profile, device, source, session, progress)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def inspect(pack, source, session):
        try:
            return service.inspect(pack, source, session)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Tab("音色包生成", id="voices"):
        source = gr.Textbox(label="源模型目录", value=source_model_dir)
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
        reference.change(preview, reference, [audio, info])
        built = make.click(build, [reference, vid, name, gender, profile, device, source, session], [output, report], concurrency_limit=None)
        unload.click(service.unload_producer, outputs=status, concurrency_id="producer", concurrency_limit=1)
        library_voice, library_profile = library_controls(library, service.rebuild, session, source, device)
        built.success(lambda v: gr.update(choices=choices(library), value=v), vid, library_voice)
        with gr.Accordion("检查已有音色包", open=False):
            pack = gr.File(label="上传 .ivp", file_types=[".ivp"], type="filepath")
            gr.Markdown("使用上方源模型目录检查兼容性；留空目录时仅检查包结构和完整性。")
            check = gr.Button("检查音色包")
            inspected = gr.JSON(label="包检查结果")
            check.click(inspect, [pack, source, session], inspected)
    return SimpleNamespace(voice=library_voice, profile=library_profile)
