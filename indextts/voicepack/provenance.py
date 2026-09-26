"""Reproducible producer fingerprints, without loading reference models."""

import hashlib
import json
from pathlib import Path

PREPROCESS = {
    "version": "indextts2.5-reference-v1",
    "load": "librosa-default-mono-22050",
    "maxSeconds": 15,
    "resampleRates": [22050, 16000],
    "fbank": {"melBins": 80, "dither": 0, "sampleFrequency": 16000},
    "basisSelection": "per-class-cosine-argmax",
}
PREPROCESS_FINGERPRINT = hashlib.sha256(
    json.dumps(PREPROCESS, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_fingerprint(model_dir, cfg):
    root = Path(model_dir)
    encoder = root / "hf_cache" / "w2v-bert-2.0"
    paths = sorted(path for path in encoder.iterdir()
                   if path.is_file() and (path.name in {"config.json", "preprocessor_config.json"}
                                          or path.suffix in {".bin", ".safetensors"}
                                          or path.name.endswith(".index.json")))
    if not any(path.suffix in {".bin", ".safetensors"} for path in paths):
        raise FileNotFoundError(f"Missing reference encoder weights: {encoder}")
    paths += [root / "hf_cache" / "campplus_cn_common.bin",
              root / str(cfg.w2v_stat), root / str(cfg.emo_matrix), root / str(cfg.spk_matrix)]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()
