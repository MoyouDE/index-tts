"""CPU emotion page; optional session bridge carries only a vector copy."""
import gradio as gr
import pandas as pd


def build_page(service, session, *, emotion_model_dir, bridge=None):
    bridge = bridge if bridge is not None else gr.State(None)
    labels = ["高兴", "愤怒", "悲伤", "恐惧", "厌恶", "低落", "惊讶", "平静"]
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
            return summary, *plots, result, path, result["context"]["rendered_text"], list(result["vector"])
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Tab("情感推理", id="emotion"):
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
                  [summary, raw_plot, final_plot, result, download, rendered, bridge], concurrency_id="emotion", concurrency_limit=1)
        release = gr.Button("卸载情感模型")
        emo_status = gr.Textbox(label="情感模型状态", interactive=False)
        release.click(service.unload_emotion, outputs=emo_status, concurrency_id="emotion")
