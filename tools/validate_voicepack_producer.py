"""Compare the standalone producer with the full TTS reference in fresh processes.

Run from the repository root: python tools/validate_voicepack_producer.py --output outputs/producer-parity
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--worker", choices=["full", "reference"])
    parser.add_argument("--profile", default="compatible-fp32")
    parser.add_argument("--compare-only", action="store_true", help="Recheck existing outputs without running models")
    args = parser.parse_args()
    import torch
    from indextts.voicepack.builder import VoicePackBuilder
    from indextts.voicepack.archive import load_voicepack
    torch.set_num_threads(4)
    if args.worker:
        destination = args.output / args.profile / args.worker
        destination.mkdir(parents=True, exist_ok=False)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        tts = None
        if args.worker == "full":
            from indextts.infer_v2_5 import IndexTTS2
            tts = IndexTTS2(model_dir=args.model_dir, cfg_path=str(Path(args.model_dir) / "config.yaml"),
                            device="cuda:0", use_bf16=args.profile == "fixed-voice-bf16",
                            use_cuda_kernel=False, use_qwen_emo=False)
        builder = VoicePackBuilder(tts, model_dir=args.model_dir, device="cuda:0", profile=args.profile)
        for index in (1, 2):
            builder.build(ROOT / f"examples/voice_0{index}.wav",
                          {"voiceId": f"parity-{index}", "displayName": f"parity-{index}", "gender": "unknown"},
                          destination / f"voice-{index}.ivp")
        if args.worker == "reference":
            forbidden = ("indextts.infer_v2_5", "indextts.gpt.model_v2", "indextts.codec.models",
                         "indextts.s2mel.modules.flow_matching", "indextts.s2mel.modules.bigvgan.bigvgan")
            assert not any(name in sys.modules for name in forbidden), "Generation module imported"
        result = {"seconds": time.perf_counter() - started,
                  "peakAllocatedBytes": torch.cuda.max_memory_allocated(),
                  "peakReservedBytes": torch.cuda.max_memory_reserved()}
        (destination / "run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if not args.compare_only:
        args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for profile in ("compatible-fp32", "fixed-voice-bf16"):
        for mode in ("full", "reference"):
            if args.compare_only:
                continue
            with (args.output / f"{profile}-{mode}.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
                                "--profile", profile, "--output", str(args.output),
                                "--model-dir", args.model_dir], check=True, stdout=log, stderr=subprocess.STDOUT)
        for index in (1, 2):
            before = load_voicepack(args.output / profile / "full" / f"voice-{index}.ivp")
            after = load_voicepack(args.output / profile / "reference" / f"voice-{index}.ivp")
            tensors = {}
            for name, value in before.tensors.items():
                actual = after.tensors[name]
                tensors[name] = {"equal": value.dtype == actual.dtype and value.shape == actual.shape and torch.equal(value, actual),
                                 "shape": list(actual.shape), "dtype": str(actual.dtype),
                                 "maxAbsError": float((value.float() - actual.float()).abs().max()),
                                 "sha256": hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            result = {"profile": profile, "reference": f"voice_0{index}.wav", "tensors": tensors,
                      "sourceModelFingerprint": after.manifest["sourceModelFingerprint"],
                      "provenance": after.manifest["provenance"],
                      "manifestEqual": before.manifest == after.manifest}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {"comparisons": results, "runs": {
        f"{profile}/{mode}": json.loads((args.output / profile / mode / "run.json").read_text())
        for profile in ("compatible-fp32", "fixed-voice-bf16") for mode in ("full", "reference")}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert all(r["manifestEqual"] and all(t["equal"] for t in r["tensors"].values()) for r in results)


if __name__ == "__main__":
    main()
