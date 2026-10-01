"""Real production/emotion/audition compatibility checks with supplied local assets."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "baseline-pack", "source-model", "emotion-model", "reader-model", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--voice-id", default="decoupling-" + uuid.uuid4().hex)
    parser.add_argument("--expected-wav-sha256")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    from indextts.workbench_service import WorkbenchService
    from indextts.voicepack.archive import load_voicepack
    from indextts.voicepack.provenance import sha256_file
    from indextts.runtime.emotion import OnnxEmotionProvider
    import torch

    old = load_voicepack(args.baseline_pack)
    profile = old.manifest["provenance"]["profile"]
    service = WorkbenchService(args.output.parent/"sessions", args.workspace)
    session = uuid.uuid4().hex
    try:
        path, production = service.build(args.reference, args.voice_id, "解耦验证音色", "unknown",
            profile, "cuda:0", args.source_model, session)
        new = load_voicepack(path)
        comparisons = {name: {"shapeEqual": old.tensors[name].shape == tensor.shape,
            "dtypeEqual": old.tensors[name].dtype == tensor.dtype, "valuesEqual": torch.equal(old.tensors[name], tensor)}
            for name, tensor in new.tensors.items()}
        assert all(all(comparison.values()) for comparison in comparisons.values())
        assert old.manifest["provenance"] == new.manifest["provenance"]
        print("Real production: exact tensors and provenance preserved", flush=True)
        rows = [["s1", "chapter", 0, "narration", "他终于平安回来了。"],
                ["s2", "chapter", 0, "dialogue", "太好了，我一直在等你！"]]
        emotion, _ = service.analyze(args.emotion_model, "上下文", "", "dialogue", rows, "s2", True, .355, session)
        baseline = OnnxEmotionProvider(args.emotion_model).analyze_window_details(emotion["input"]["sentences"], "s2")
        for field in ("rawVector", "vector", "totalIntensity", "threshold", "baseFallback", "context"):
            assert emotion[field] == baseline[field], field
        print("Real emotion: provider outputs and context unchanged", flush=True)
        wav, _, audition = service.audition(args.voice_id, profile, args.reader_model, "cuda:0",
            "今天阳光很好。", "base", 1, 17, {}, "all", 4, "native", session)
        digest = sha256_file(wav)
        if args.expected_wav_sha256:
            assert digest == args.expected_wav_sha256, (digest, args.expected_wav_sha256)
        report = {"production": production, "tensorComparisonWithPreviousPackage": comparisons,
            "provenanceEqual": True, "emotion": emotion, "audition": audition, "wavSha256": digest}
        if args.expected_wav_sha256:
            report["matchesPreviousFixedSeedWav"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Real audition: completed with seed 17", flush=True)
    finally:
        service.close()


if __name__ == "__main__":
    main()
