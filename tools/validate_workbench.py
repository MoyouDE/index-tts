"""Real sidecar regression: legacy/default settings and changed acoustic controls."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indextts.audition_worker import AuditionWorker
from indextts.runtime.profiles import FP32, BF16, generation_options
from indextts.voicepack.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", required=True)
    parser.add_argument("--bf16-model", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for profile, model in ((FP32, args.fp32_model), (BF16, args.bf16_model)):
        with tempfile.TemporaryDirectory(dir=args.output.parent) as cache:
            worker = AuditionWorker(Path(model).resolve(), "cuda:0", cache, "all", 4, "native")
            try:
                health = worker.call("health")
                voice = args.workspace.resolve()/"voices"/args.voice_id/(profile+".ivp")
                worker.call("voices.load", {"path": str(voice)})
                request = {"voiceId": args.voice_id, "text": "今天阳光很好。", "seed": 17, "emotion": "base"}
                options = {k: v for k, v in generation_options(profile).items() if k != "num_return_sequences"}
                options.update(acoustic_steps=25, cfg=.7)
                runs = []
                for label, overrides in (("legacy", None), ("explicit-defaults", options),
                                         ("changed-acoustic", {**options, "acoustic_steps": 10, "cfg": .5})):
                    params = {**request}
                    if overrides is not None:
                        params["generationSettings"] = overrides
                    result = worker.call("synthesize", params)
                    path = Path(result.pop("audioPath"))
                    runs.append({"case": label, "wavSha256": sha256_file(path), "result": result})
                    path.unlink()
                assert runs[0]["wavSha256"] == runs[1]["wavSha256"]
                assert runs[1]["wavSha256"] != runs[2]["wavSha256"]
                assert runs[2]["result"]["generationSettings"]["acoustic_steps"] == 10
                report[profile] = {"legacyEqualsExplicitDefaults": True, "acousticChangeAffectsOutput": True,
                                   "packSha256": sha256_file(voice), "health": health, "runs": runs}
                print(f"Passed {profile}", flush=True)
            finally:
                worker.close()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
