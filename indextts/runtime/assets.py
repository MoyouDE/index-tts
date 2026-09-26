"""Fetch/verify the exact optimization reference assets using a tracked hash lock."""

import argparse
import json
import os
import uuid
from pathlib import Path

from indextts.voicepack.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "tests/fixtures/reader-assets.lock.json"


def sources():
    main = "https://huggingface.co/IndexTeam/IndexTTS-2.5/resolve/c39ce5ba981572cb187443877ff559dfb246ce63/"
    w2v = "https://huggingface.co/facebook/w2v-bert-2.0/resolve/da985ba0987f70aaeb84a80f2851cfac8c697a7b/"
    vocoder = "https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x/resolve/633ff708ed5b74903e86ff1298cf4a98e921c513/"
    result = {f"checkpoints/{name}": main + name for name in
              ("config.yaml", "gpt.pth", "s2mel.pth", "codec.pth", "wav2vec2bert_stats.pt",
               "feat1.pt", "feat2.pt", "multilingual_zh_ja_yue_char_del.tiktoken")}
    result.update({f"checkpoints/hf_cache/w2v-bert-2.0/{name}": w2v + name for name in
                   ("config.json", "preprocessor_config.json", "model.safetensors")})
    result.update({f"checkpoints/hf_cache/bigvgan/{name}": vocoder + name
                   for name in ("config.json", "bigvgan_generator.pt")})
    result["checkpoints/hf_cache/campplus_cn_common.bin"] = (
        "https://huggingface.co/funasr/campplus/resolve/e4b6ede7ce16997aff4ae69fbca1f0175e2afede/campplus_cn_common.bin")
    example = "https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/b01840e8e4fd9753743a6d0466cd73ae1d634a68/examples/"
    result.update({f"examples/{name}": example + name for name in ("voice_01.wav", "voice_02.wav")})
    return result


def asset_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Asset path escapes root: {name}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["record", "verify", "fetch"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "record":
        if args.lock.exists():
            raise FileExistsError(args.lock)
        files = {}
        for name, url in sources().items():
            path = asset_path(root, name)
            files[name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps({"schemaVersion": 1, "files": files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schemaVersion") != 1:
        raise ValueError("Unsupported asset lock")
    for name, spec in lock["files"].items():
        path = asset_path(root, name)
        if path.is_file():
            if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
                raise ValueError(f"Existing asset mismatch (not overwritten): {name}")
        elif args.command == "verify":
            raise FileNotFoundError(path)
        else:
            import requests
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.download")
            try:
                with requests.get(spec["url"], stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with temporary.open("xb") as stream:
                        count = 0
                        for chunk in response.iter_content(1024 * 1024):
                            count += len(chunk)
                            if count > spec["bytes"]:
                                raise ValueError(f"Remote asset exceeds locked size: {name}")
                            stream.write(chunk)
                if temporary.stat().st_size != spec["bytes"] or sha256_file(temporary) != spec["sha256"]:
                    raise ValueError(f"Remote asset does not match locked bytes: {name}")
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        print(f"verified {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
