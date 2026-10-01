"""Local validation entrypoint with independently selectable feature pages."""
import argparse
import logging
import uuid
from pathlib import Path
from .web_modules import parse_modules

DEFAULT_SOURCE_MODEL_DIR = str(Path(__file__).resolve().parents[1] / "voice-producer" / "models" / "checkpoints")


def create_app(*, source_model_dir=DEFAULT_SOURCE_MODEL_DIR, emotion_model_dir="", output_dir="outputs/validation-web",
               workspace_dir="outputs/voice-workbench", modules=None, cpu_threads=4):
    enabled = parse_modules(modules)
    if not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    import gradio as gr
    from .workbench_service import WorkbenchService
    service = WorkbenchService(output_dir, workspace_dir, modules=enabled, cpu_threads=cpu_threads)
    producer_only = enabled == ("producer",)
    title = "IndexTTS 音色生成工具" if producer_only else "IndexTTS 模块验证"
    description = ("上传参考音频，生成、管理和下载本地音色包。每种精度保留最新包，不写入交接目录。"
                   if producer_only else "本地音色制包、管理、情感验证与合成试听。工作区保留每种精度的最新包和最近一次试听，不写入交接目录。")
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as app:
        session = gr.State(lambda: uuid.uuid4().hex)
        bridge = gr.State(None) if {"emotion", "audition"}.issubset(enabled) else None
        gr.Markdown(f"# {title}\n{description}")
        pages = {}
        with gr.Tabs() as tabs:
            if "producer" in enabled:
                from .producer_web import build_page
                pages["producer"] = build_page(service, service.library, session, source_model_dir=source_model_dir)
            if "emotion" in enabled:
                from .emotion_web import build_page
                build_page(service, session, emotion_model_dir=emotion_model_dir, bridge=bridge)
            if "audition" in enabled:
                from .audition_web import build_page
                pages["audition"] = build_page(service, service.library, session, emotion_bridge=bridge,
                                               producer_enabled="producer" in enabled)
        if {"producer", "audition"}.issubset(enabled):
            from .web_common import choices
            audition, producer = pages["audition"], pages["producer"]
            audition.missing.click(lambda v, p: (gr.update(selected="voices"),
                gr.update(choices=choices(service.library), value=v), p),
                [audition.voice, audition.profile], [tabs, producer.voice, producer.profile])
        if pages:
            from .web_common import choices
            selectors = [page.voice for page in pages.values()]
            app.load(lambda: tuple(gr.update(choices=choices(service.library)) for _ in selectors)
                     if len(selectors) > 1 else gr.update(choices=choices(service.library)), outputs=selectors)
    # Public application owner for explicit shutdown in tests and embedding callers.
    app.workbench = service
    app.queue(default_concurrency_limit=1)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", default=DEFAULT_SOURCE_MODEL_DIR,
                        help="制包源权重固定目录；仅在启动时配置，页面只读")
    parser.add_argument("--emotion-model-dir", default="")
    parser.add_argument("--output-dir", default="outputs/validation-web")
    parser.add_argument("--workspace-dir", default="outputs/voice-workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7861, type=int)
    parser.add_argument("--cpu-threads", default=4, type=int)
    parser.add_argument("--modules", type=parse_modules, default=parse_modules(), help="逗号分隔: producer,emotion,audition；默认全部")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_app(source_model_dir=args.source_model_dir, emotion_model_dir=args.emotion_model_dir, output_dir=args.output_dir, workspace_dir=args.workspace_dir, modules=args.modules, cpu_threads=args.cpu_threads)
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=True,
               max_file_size="100mb", allowed_paths=[str(Path(args.workspace_dir).expanduser().resolve())] if {"producer", "audition"}.intersection(args.modules) else [])


if __name__ == "__main__":
    main()
