"""CLI entry point for the reader runtime and JSONL sidecar."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indextts-runtime", description="IndexTTS reader runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)

    export = subcommands.add_parser("export-model", help="Export reference-free runtime weights")
    export.add_argument("--source-model-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--config")

    doctor = subcommands.add_parser("doctor", help="Validate runtime files and CUDA support")
    doctor.add_argument("--model-dir", required=True)
    doctor.add_argument("--device", default="cuda:0")

    serve = subcommands.add_parser("serve", help="Run the JSONL stdio sidecar")
    serve.add_argument("--model-dir", required=True)
    serve.add_argument("--voice-dir", action="append", required=True)
    serve.add_argument("--cache-dir", required=True)
    serve.add_argument("--device", default="cuda:0")
    serve.add_argument("--emotion-backend", choices=["qwen", "explicit"], default="qwen")
    serve.add_argument("--qwen-model-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export-model":
        from .model_export import export_runtime_model

        manifest = export_runtime_model(args.source_model_dir, args.output_dir, cfg_path=args.config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        import torch

        from .model_export import verify_runtime_model

        manifest = verify_runtime_model(args.model_dir)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        device = torch.device(args.device)
        result = {
            "ok": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "totalVramBytes": torch.cuda.get_device_properties(device).total_memory,
            "coreModelBytes": manifest["coreModelBytes"],
            "sourceModelFingerprint": manifest["sourceModelFingerprint"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        from .emotion import ExplicitEmotionProvider, QwenEmotionProvider
        from .engine import ReaderRuntime
        from .sidecar import JsonlSidecar

        if args.emotion_backend == "qwen":
            if not args.qwen_model_dir:
                raise ValueError("emotion-backend=qwen 时必须提供 --qwen-model-dir")
            provider = QwenEmotionProvider(args.qwen_model_dir)
        else:
            provider = ExplicitEmotionProvider([0, 0, 0, 0, 0, 0, 0, 0.8])
        with contextlib.redirect_stdout(sys.stderr):
            runtime = ReaderRuntime(
                args.model_dir,
                args.voice_dir,
                provider,
                args.device,
                cache_dir=args.cache_dir,
            )
        return JsonlSidecar(runtime).serve()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
