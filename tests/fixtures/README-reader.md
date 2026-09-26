# Reader inference fixtures

- `reader-benchmark.json`: 20 original Chinese passages in five length bins; punctuation counts toward length. `full` × two voices × four emotions × three seeds gives 480 cases.
- `reader-assets.lock.json`: exact source weights, configs and two public demo recordings. All small files were checked against their pinned URLs; all large weights against Hugging Face LFS SHA-256 metadata. Run `python -m indextts.runtime.assets fetch` to rebuild the source tree without the parent repository.
- `voices-bf16/`: two schema-v2 packs actually produced by the BF16 reference path from the locked `voice_01.wav` and `voice_02.wav`. Each archive contains producer/version/source fingerprints and licenses. These are not FP32-to-BF16 conversions.
- `voices-fp32/`: two existing schema-v1 FP32 compatibility packs copied verbatim from the parent repository's tracked SoundPackage. Their source audio provenance was not stored historically and is not invented here. Both match the locked main generation-model fingerprint.

No source recording or generation weights are included in the voice archives. Preserve the embedded licenses and derivative notices. The fixtures validate the implementation, not the perceptual quality of the original model.
