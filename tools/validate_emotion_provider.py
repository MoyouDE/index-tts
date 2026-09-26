"""Exercise the real CPU ONNX provider and save reproducible input/output traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indextts.runtime.emotion import OnnxEmotionProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provider = OnnxEmotionProvider(args.model_dir)
    provider._load()

    def sentence(sid, text, kind="dialogue", line=0):
        return dict(sentenceId=sid, text=text, sentenceType=kind, sectionId="chapter", lineIndex=line)

    cases = {
        "single": [sentence("target", "太好了，我们终于成功了！")],
        "context": [sentence("previous", "暴风雨过去，他终于平安回来了。", "narration"),
                    sentence("target", "太好了，我一直在等你！"),
                    sentence("next", "她激动地迎了上去。", "narration")],
        "long-target": [sentence("target", "他终于平安回来了。" * 100)],
    }
    results = {}
    for name, sentences in cases.items():
        details = provider.analyze_window_details(sentences, "target")
        assert details["vector"] == provider.analyze_window(sentences, "target")
        assert details["context"]["input_token_count"] <= 512
        results[name] = {"input": sentences, "details": details, "legacyEqual": True}
    fallback = provider.analyze_window_details(cases["context"], "target", neutral_threshold=1.0)
    assert fallback["baseFallback"] and fallback["vector"] == [0.0] * 8
    assert provider.neutral_threshold == provider.model_threshold
    results["temporary-threshold"] = fallback
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"modelManifest": provider.model_manifest,
                   "manifestSha256": hashlib.sha256((args.model_dir / "emotion_model.json").read_bytes()).hexdigest(),
                   "executionProviders": provider._session.get_providers(),
                   "cases": results}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Passed real ONNX comparisons: {args.output}")


if __name__ == "__main__":
    main()
