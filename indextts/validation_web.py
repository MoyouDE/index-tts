"""Independent local voice-pack and ONNX emotion validation Web UI."""
import argparse
import uuid


def create_app(*, source_model_dir="checkpoints", emotion_model_dir="", output_dir="outputs/validation-web"):
    import gradio as gr
    import pandas as pd
    from .validation_service import ValidationService, audio_details
    service = ValidationService(output_dir)
    labels = ["高兴", "愤怒", "悲伤", "恐惧", "厌恶", "低落", "惊讶", "平静"]

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

    def reset_threshold(model_dir):
        try:
            threshold, manifest = service.emotion_default(model_dir)
            return True, threshold, manifest
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def analyze(*args):
        try:
            result, path = service.analyze(*args)
            plots = [pd.DataFrame({"情感": labels, "分数": result[key]})
                     for key in ("rawVector", "vector")]
            summary = (f"总强度 **{result['totalIntensity']:.4f}** · 阈值 **{result['threshold']:.4f}** · "
                       f"{'使用基础情感（零向量）' if result['baseFallback'] else '保留模型情感向量'}")
            return summary, *plots, result, path, result["context"]["rendered_text"]
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="IndexTTS 模块验证", theme=gr.themes.Soft()) as app:
        session = gr.State(lambda: uuid.uuid4().hex)
        gr.Markdown("# IndexTTS 模块验证\n独立检查音色包生成与文本情感推理。生成的音色包可交给阅读器使用；本页不合成试听音频。")
        with gr.Tab("音色包生成"):
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
            report = gr.JSON(label="生成与校验结果")
            reference.change(preview, reference, [audio, info])
            make.click(build, [reference, vid, name, gender, profile, device, source, session], [output, report], concurrency_id="producer", concurrency_limit=1)
            unload.click(service.unload_producer, outputs=status, concurrency_id="producer", concurrency_limit=1)
            with gr.Accordion("检查已有音色包", open=False):
                pack = gr.File(label="上传 .ivp", file_types=[".ivp"], type="filepath")
                gr.Markdown("使用上方源模型目录检查兼容性；留空目录时仅检查包结构和完整性。")
                check = gr.Button("检查音色包")
                inspected = gr.JSON(label="包检查结果")
                check.click(inspect, [pack, source, session], inspected)
        with gr.Tab("情感推理"):
            model_dir = gr.Textbox(label="情感 ONNX 模型目录", value=emotion_model_dir,
                                   placeholder="包含 emotion.onnx、emotion_model.json 和 tokenizer 文件的目录")
            mode = gr.Radio(["单句", "上下文"], value="单句", label="输入模式")
            with gr.Group() as single:
                text = gr.Textbox(label="目标文本", lines=3)
                kind = gr.Radio([("对白", "dialogue"), ("旁白", "narration")], value="dialogue", label="句子类型")
            with gr.Group(visible=False) as window:
                rows = gr.Dataframe(headers=["句子 ID", "章节", "段落", "类型", "正文"],
                    datatype=["str", "str", "number", "str", "str"], type="array", col_count=(5, "fixed"),
                    value=[["s1", "chapter-1", 0, "旁白", "他终于平安回来了。"],
                           ["s2", "chapter-1", 0, "对白", "太好了，我一直在等你！"]],
                    label="上下文句子（同一章节、按原文顺序；类型填写对白或旁白）")
                target = gr.Textbox(label="目标句 ID", value="s2")
            mode.change(lambda value: (gr.update(visible=value == "单句"), gr.update(visible=value == "上下文")), mode, [single, window])
            with gr.Row():
                use_default = gr.Checkbox(value=True, label="使用模型默认阈值")
                threshold = gr.Slider(0, 1, value=0.355, step=0.001, label="临时阈值（取消默认选项后生效）")
                reset = gr.Button("加载模型／恢复默认阈值")
            with gr.Accordion("模型信息", open=False):
                manifest = gr.JSON(label="情感模型 manifest")
            reset.click(reset_threshold, model_dir, [use_default, threshold, manifest], concurrency_id="emotion")
            run = gr.Button("分析情感", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                with gr.Column(min_width=340):
                    raw_plot = gr.BarPlot(x="情感", y="分数", y_lim=[0, 1], sort=labels, label="模型原始八维输出")
                with gr.Column(min_width=340):
                    final_plot = gr.BarPlot(x="情感", y="分数", y_lim=[0, 1], sort=labels, label="阈值处理后八维向量")
            rendered = gr.Textbox(label="实际模型输入", lines=4, interactive=False)
            download = gr.File(label="下载结果 JSON", interactive=False)
            with gr.Accordion("详细结果与上下文选择", open=False):
                result = gr.JSON(label="推理详情")
            run.click(analyze, [model_dir, mode, text, kind, rows, target, use_default, threshold, session],
                      [summary, raw_plot, final_plot, result, download, rendered], concurrency_id="emotion", concurrency_limit=1)
            release = gr.Button("卸载情感模型")
            emo_status = gr.Textbox(label="情感模型状态", interactive=False)
            release.click(service.unload_emotion, outputs=emo_status, concurrency_id="emotion")
    app.queue(default_concurrency_limit=1)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", default="checkpoints")
    parser.add_argument("--emotion-model-dir", default="")
    parser.add_argument("--output-dir", default="outputs/validation-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7861, type=int)
    parser.add_argument("--cpu-threads", default=4, type=int)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    import torch
    torch.set_num_threads(args.cpu_threads)
    app = create_app(source_model_dir=args.source_model_dir, emotion_model_dir=args.emotion_model_dir, output_dir=args.output_dir)
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=True,
               max_file_size="100mb")


if __name__ == "__main__":
    main()
