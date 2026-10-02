"""Producer workflow; model settings and library management stay out of its main path."""
from pathlib import Path
from types import SimpleNamespace
import uuid
import gradio as gr
from .library_web import library_controls
from .web_common import choices, ui_errors


def build_page(service, library, session, *, source_model_dir):
    source_model_dir = str(Path(source_model_dir).expanduser().resolve())

    @ui_errors
    def inspect(pack, session):
        return service.inspect(pack, source_model_dir, session)

    @ui_errors
    def build(selection, name, gender, profile, device, sid, progress=gr.Progress()):
        name = (name or "").strip()
        if not name or len(name) > 128:
            raise ValueError("请填写音色名称（1～128 个字符）")
        vid = "voice-" + uuid.uuid4().hex[:16]
        path, report = service.build_selection(selection, vid, name, gender, profile, device,
                                             source_model_dir, sid, progress)
        method = "多段身份等权融合" if selection["method"] == "speaker-mean-v1" else "单段主参考"
        message = f"已保存“{name}” · {method} · {'FP32' if profile == 'compatible-fp32' else 'BF16'} · {report['seconds']:.2f} 秒"
        return path, report, message

    with gr.Tab("音色包生成", id="voices"):
        with gr.Row():
            with gr.Column(scale=3, min_width=480):
                name = gr.Textbox(label="音色名称", placeholder="给这个声音起个名字", render=False)
                gender = gr.Dropdown([("未知", "unknown"), ("女声", "female"), ("男声", "male"), ("中性", "neutral")],
                                     value="unknown", label="性别", render=False)
                profile = gr.Radio([("FP32", "compatible-fp32"), ("BF16", "fixed-voice-bf16")],
                                   value="compatible-fp32", label="制包精度", render=False)
                device = gr.Dropdown([("自动", "auto"), ("CPU", "cpu"), ("CUDA 0", "cuda:0")], value="auto",
                                     allow_custom_value=True, label="计算设备", render=False)
                method = gr.Radio([("自动：一段用单段，多段用等权融合", "auto"),
                                   ("只用主参考", "primary-only-v1"), ("多段身份等权融合", "speaker-mean-v1")],
                                  value="auto", label="参考构建方法", render=False)
                output = gr.File(label="下载生成的音色包", interactive=False, render=False)
                report = gr.JSON(label="制包详情", render=False)
                result = gr.Textbox(label="生成结果", interactive=False, render=False)

                def diagnostics():
                    gr.Textbox(label="源模型目录", value=source_model_dir, interactive=False)
                    gr.Markdown("模型位置由工具启动配置固定。默认 FP32、自动设备；BF16 需匹配的推理模型。融合是否适合该素材，仍需试听判断。")
                    unload = gr.Button("卸载制包模型")
                    model_status = gr.Textbox(label="模型状态", interactive=False)
                    unload.click(service.unload_producer, outputs=model_status, concurrency_id="producer", concurrency_limit=1)
                    report.render()
                    with gr.Accordion("检查已有音色包", open=False):
                        pack = gr.File(label="上传 .ivp", file_types=[".ivp"], type="filepath")
                        check = gr.Button("检查音色包")
                        inspected = gr.JSON(label="包检查结果")
                        check.click(inspect, [pack, session], inspected)

                from .material_web import build_controls
                material_page = build_controls(service.materials, session, generate=build,
                    session_directory=service.session_dir, inputs=[name, gender, profile, device, session],
                    outputs=[output, report, result], name=name, options=[profile, device, gender, method],
                    method=method, diagnostics=diagnostics)
            with gr.Column(scale=1, min_width=280):
                library_voice, library_profile = library_controls(library, service.rebuild, session, source_model_dir, device)
        def select_generated(result):
            manifest = result["manifest"]
            return (gr.update(choices=choices(library), value=manifest["voiceId"]),
                    manifest["provenance"]["profile"])
        # File inputs are Gradio cache copies, so identity must come from metadata.
        material_page.built.success(select_generated, report, [library_voice, library_profile])
    return SimpleNamespace(voice=library_voice, profile=library_profile, materials=material_page)
