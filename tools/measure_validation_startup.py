"""Measure feature-page construction in a fresh process, without model loading."""
import argparse
import json
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("producer,emotion,audition", "producer", "emotion", "audition", "producer,audition", "emotion,audition")
WATCHED = ("torch", "transformers", "onnxruntime", "indextts.voicepack.builder",
           "indextts.voicepack.archive", "indextts.voicepack.reference", "indextts.runtime.engine",
           "indextts.producer_service", "indextts.emotion_service", "indextts.audition_service",
           "indextts.producer_web", "indextts.emotion_web", "indextts.audition_web")


def measure(mode, baseline=False, serve=False):
    import psutil
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        if baseline:
            # Overlay only the previous entry/service/UI files; all unchanged
            # runtime dependencies resolve from this checkout.
            package = Path(directory) / "indextts"
            package.mkdir()
            (package / "__init__.py").write_text(
                "__path__.append(" + repr(str(ROOT / "indextts")) + ")\n", encoding="utf-8")
            for name in ("validation_web.py", "validation_service.py", "workbench_service.py", "workbench_web.py"):
                content = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"4a9f814:indextts/{name}"],
                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
                (package/name).write_text(content, encoding="utf-8")
            sys.path.insert(0, directory)
        from indextts.validation_web import create_app
        work = Path(directory) / "work"
        options = {} if baseline else {"modules": mode}
        app = create_app(**options, output_dir=Path(directory)/"out", workspace_dir=work,
                         source_model_dir="missing", emotion_model_dir="missing")
        construction_seconds = time.perf_counter()-started
        if serve:
            with socket.socket() as port_picker:
                port_picker.bind(("127.0.0.1", 0))
                port = port_picker.getsockname()[1]
            app.launch(server_name="127.0.0.1", server_port=port, share=False,
                       prevent_thread_lock=True, quiet=True)
        result = {"modules": mode, "constructionSeconds": round(time.perf_counter()-started, 3),
                  "rssBytes": psutil.Process().memory_info().rss,
                  "rssScope": "当前 Python 进程工作集；不含子进程，不是显存",
                  "loadedModules": {name: name in sys.modules for name in WATCHED},
                  "workspaceCreated": work.exists(),
                  "tabs": [c["props"]["label"] for c in app.config["components"] if c["type"] == "tabitem"]}
        result["constructionSeconds"] = round(construction_seconds, 3)
        if serve:
            result["serverReadySeconds"] = round(time.perf_counter()-started, 3)
            app.close()
        if not baseline:
            app.workbench.close()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", action="store_true", help="Previous committed combined page (4a9f814)")
    parser.add_argument("--serve", action="store_true", help="Also start a temporary localhost server and measure readiness")
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(measure(args.mode, args.baseline, args.serve), ensure_ascii=False))
        return
    results = []
    for mode in MODES[:1] if args.baseline else MODES:
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode]
        if args.baseline:
            command.append("--baseline")
        if args.serve:
            command.append("--serve")
        completed = subprocess.run(command,
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180, check=True)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append(result)
        print(f"{mode}: {result['constructionSeconds']}s, RSS {result['rssBytes']/1024**2:.1f} MiB", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
