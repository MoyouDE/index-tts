"""Command line interface for creating and inspecting IndexTTS voice packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .archive import load_voicepack
from .builder import VoicePackBuilder, model_fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indextts-voicepack", description="Build and validate IndexTTS .ivp voice packs")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="Build a voice pack from reference audio")
    build.add_argument("--reference", required=True)
    build.add_argument("--voice-id", required=True)
    build.add_argument("--name", required=True)
    build.add_argument("--gender", choices=["female", "male", "neutral", "unknown"], default="unknown")
    build.add_argument("--output", required=True)
    build.add_argument("--model-dir", default="checkpoints")
    build.add_argument("--config")
    build.add_argument("--device")
    build.add_argument("--fp16", action="store_true", help="Use BF16 on supported GPUs")

    inspect = subcommands.add_parser("inspect", help="Print a pack manifest")
    inspect.add_argument("pack")

    verify = subcommands.add_parser("verify", help="Verify pack integrity and source model compatibility")
    verify.add_argument("pack")
    verify.add_argument("--model-dir", required=True)
    verify.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        builder = VoicePackBuilder(
            model_dir=args.model_dir,
            cfg_path=args.config,
            device=args.device,
            use_bf16=args.fp16,
        )
        output = builder.build(
            args.reference,
            {"voiceId": args.voice_id, "displayName": args.name, "gender": args.gender},
            args.output,
        )
        print(output)
        return 0
    if args.command == "inspect":
        pack = load_voicepack(args.pack)
        print(json.dumps(pack.manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        fingerprint = model_fingerprint(args.model_dir, args.config)
        pack = load_voicepack(args.pack, expected_model_fingerprint=fingerprint)
        print(json.dumps({"ok": True, "voiceId": pack.voice_id, "path": str(Path(args.pack).resolve())}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
