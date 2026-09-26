"""Reproducible same-profile baseline/optimized benchmark; run each in a fresh process."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--voice-dir", action="append", required=True)
    parser.add_argument("--voice-id", action="append", required=True)
    parser.add_argument("--output", required=True, help="JSONL output; refuses to overwrite")
    parser.add_argument("--cases", default=str(Path(__file__).resolve().parents[2] / "tests/fixtures/reader-benchmark.json"))
    parser.add_argument("--optimizations", default="all")
    parser.add_argument("--suite", choices=["smoke", "full", "voices", "alternating"], default="smoke")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--compare", help="Baseline JSONL; exact conditioning/token/PCM digest comparison")
    parser.add_argument("--trace", action="store_true", help="Correctness run (excluded from formal timing claims)")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allocator", choices=["native", "cudaMallocAsync"])
    parser.add_argument("--performance-cores", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.cpu_threads < 1:
        parser.error("rounds and cpu-threads must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if args.compare and not args.trace:
        parser.error("--compare requires --trace")
    if args.allocator:
        if "torch" in sys.modules:
            raise RuntimeError("Allocator must be configured before importing torch")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"backend:{args.allocator}"
    import torch
    import transformers
    from .engine import ReaderRuntime
    from .profiles import InferenceOptimizations
    from .gpu_memory import ProcessGpuMemorySampler
    from indextts.voicepack.provenance import sha256_file

    torch.set_num_threads(args.cpu_threads)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    selected = cases if args.suite == "full" else [cases[0], cases[4], cases[8]]
    emotions = ["base", [0.8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0.8, 0, 0, 0, 0, 0],
                [0.2, 0, 0.3, 0, 0, 0, 0.2, 0.1]]
    seeds = [17, 29, 43] if args.suite == "full" else [17]
    work = [(case, voice, emotion, seed) for case in selected for voice in args.voice_id
            for emotion in (emotions if args.suite == "full" else ["base"]) for seed in seeds]
    if args.suite == "voices":
        work = [(cases[0], voice, "base", 17) for voice in args.voice_id]
    if args.suite == "alternating":
        work = [(cases[0], args.voice_id[index % len(args.voice_id)], emotions[index % 4], 17)
                for index in range(100)]
    baseline = {}
    baseline_environment = None
    if args.compare:
        baseline_rows = [json.loads(line) for line in Path(args.compare).read_text(encoding="utf-8").splitlines()]
        baseline = {row["caseKey"]: row for row in baseline_rows if row.get("kind") == "request"}
        baseline_environment = next((row for row in baseline_rows if row.get("kind") == "environment"), None)
        if not baseline or not baseline_environment or any(not row.get("trace") for row in baseline.values()):
            parser.error("Comparison requires a non-empty baseline with traces and environment metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("x", encoding="utf-8")
    def record(row):
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    sampler = ProcessGpuMemorySampler().start()
    started = time.perf_counter()
    try:
        runtime = ReaderRuntime(args.model_dir, args.voice_dir, None, args.device,
                                cache_dir=output.parent / "audio",
                                optimizations=InferenceOptimizations.parse(args.optimizations),
                                prefer_performance_cores=args.performance_cores)
        torch.cuda.synchronize(device)
    finally:
        load_memory = sampler.stop()
    try:
        revision = subprocess.check_output(["git", "-c", f"safe.directory={Path(__file__).resolve().parents[2].as_posix()}",
                                            "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    environment = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                   "platform": platform.platform(), "gpu": torch.cuda.get_device_name(device),
                   "cuda": torch.version.cuda, "cpuThreads": args.cpu_threads, "revision": revision,
                   "allocator": torch.cuda.memory.get_allocator_backend(), "traceEnabled": args.trace,
                   "profile": runtime.profile, "optimizations": runtime.optimizations.to_dict(),
                   "modelManifestSha256": sha256_file(Path(args.model_dir) / "runtime_model.json"),
                   "voiceHashes": {voice: sha256_file(runtime.snapshot_voice(voice).path) for voice in args.voice_id}}
    if baseline_environment:
        for field in ("profile", "modelManifestSha256", "voiceHashes", "torch", "transformers", "gpu"):
            if environment[field] != baseline_environment[field]:
                raise ValueError(f"Incompatible baseline: {field}")
    record({"kind": "environment", **environment, "loadMs": (time.perf_counter() - started) * 1000,
            "loadMemory": load_memory, "health": runtime.health()})
    # The first request is deliberately separate from warm measurements.
    first = runtime.synthesize(cases[0]["text"], args.voice_id[0], seed=17)
    record({"kind": "firstRequest", "result": first})
    Path(first["audioPath"]).unlink()
    rows, mismatches, repeated = [], [], {}
    try:
        for round_index in range(args.rounds):
            for index, (case, voice, emotion, seed) in enumerate(work):
                # Stage static voice conditions outside warm RTF; switching is separately measured.
                switch_start = time.perf_counter()
                runtime._voice_condition(voice, None)
                torch.cuda.synchronize(device)
                switch_ms = (time.perf_counter() - switch_start) * 1000
                trace = {}
                callback = (lambda name, tensor: trace.update({name: tensor_digest(tensor)})) if args.trace else None
                torch.cuda.reset_peak_memory_stats(device)
                sampler = ProcessGpuMemorySampler().start()
                try:
                    result = runtime.synthesize(case["text"], voice, emotion, seed=seed, _trace=callback)
                    torch.cuda.synchronize(device)
                finally:
                    memory = sampler.stop()
                key = f"{round_index}:{index}:{case['id']}:{voice}:{seed}"
                row = {"kind": "request", "caseKey": key, "round": round_index, "text": case["text"],
                       "lengthGroup": case["group"], "voiceId": voice, "emotion": emotion, "seed": seed,
                       "switchMs": switch_ms, "trace": trace, "result": result,
                       "processMemory": memory, "memory": runtime.health()}
                audio_path = Path(result["audioPath"])
                row["result"].pop("audioPath", None)
                if baseline:
                    original = baseline.get(key)
                    row["matchesBaseline"] = bool(original and original["trace"] == trace and trace)
                    if not row["matchesBaseline"]:
                        mismatches.append(key)
                repetition_key = json.dumps([case["id"], voice, emotion, seed], sort_keys=True)
                if args.trace:
                    if repetition_key in repeated and repeated[repetition_key] != trace:
                        mismatches.append(f"repeated:{key}")
                    repeated[repetition_key] = trace
                record(row)
                rows.append(row)
                # Save PCM hashes, not duplicate generated audio, in the report.
                audio_path.unlink()
                print(f"{len(rows)}/{len(work) * args.rounds} {key} RTF={result['rtf']:.3f}", flush=True)
    finally:
        stream.close()
    import numpy as np
    if baseline:
        mismatches.extend(f"missing:{key}" for key in baseline.keys() - {row["caseKey"] for row in rows})
    summary = {"environment": environment, "requests": len(rows), "mismatches": mismatches,
               "strictVramTargetCertified": False, "groups": {}}
    for group in sorted({row["lengthGroup"] for row in rows}):
        values = [row["result"]["rtf"] for row in rows if row["lengthGroup"] == group]
        summary["groups"][group] = {"n": len(values), "medianRtf": float(np.median(values)),
                                     "p95Rtf": float(np.percentile(values, 95))}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
