"""Crop unconfirmed material candidates for human review; never runs an encoder."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(workspace, material_id, manifest, output):
    import numpy as np
    import soundfile as sf
    from indextts.material_service import MaterialService, write_json
    from indextts.voicepack.provenance import sha256_file

    candidates = json.loads(Path(manifest).read_text(encoding="utf-8"))["candidates"]
    service = MaterialService(workspace)
    record = service.record(material_id)
    ordered = sorted(candidates, key=lambda c: c["start"])
    ids = [c["id"] for c in candidates]
    if not candidates or len(ids) != len(set(ids)):
        raise ValueError("候选不能为空且 ID 不得重复")
    if any(c["role"] not in ("training", "held-out") for c in candidates):
        raise ValueError("候选用途须为 training 或 held-out")
    if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("候选片段不得重叠")
    for c in candidates:
        if not c["id"].isascii() or not c["id"].replace("-", "").isalnum():
            raise ValueError("候选 ID 只能包含 ASCII 字母、数字及横线")
        if not 0 <= c["start"] < c["end"] <= record["durationSeconds"] or c["end"]-c["start"] > 15:
            raise ValueError("候选须在素材内且不超过 15 秒")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"sourceId": material_id, "sourceSha256": record["sourceSha256"],
              "audioSha256": record["audioSha256"], "materialRevision": record["revision"],
              "confirmedTarget": False, "cleanMaterialConfirmed": False,
              "processing": "Unchanged mono float samples; montage inserts 1 second between clips only.",
              "candidates": [], "montages": {}}
    try:
        groups = {"training": [], "held-out": []}
        for c in candidates:
            path = output/(c["id"]+".wav")
            service.crop(material_id, c["start"], c["end"], path)
            wave, rate = sf.read(path, dtype="float32")
            result["candidates"].append({**c, "file": path.name, "sha256": sha256_file(path),
                                         "durationSeconds": len(wave)/rate})
            groups[c["role"]].append((c["id"], wave, rate))
        for role, clips in groups.items():
            if not clips:
                continue
            rate = clips[0][2]
            parts, timings, cursor = [], [], 0
            for clip_id, wave, sample_rate in clips:
                if sample_rate != rate:
                    raise ValueError("候选采样率不一致")
                if parts:
                    parts.append(np.zeros(rate, dtype="float32"))
                    cursor += rate
                timings.append({"id": clip_id, "start": cursor/rate, "end": (cursor+len(wave))/rate})
                parts.append(wave)
                cursor += len(wave)
            path = output/(role+"-review.wav")
            sf.write(path, np.concatenate(parts), rate, subtype="FLOAT")
            result["montages"][role] = {"file": path.name, "sha256": sha256_file(path), "timings": timings}
        write_json(output/"review.json", result)
    finally:
        service.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="outputs/voice-workbench")
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.material_id, args.manifest, args.output), ensure_ascii=False, indent=2))
