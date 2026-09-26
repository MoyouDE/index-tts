"""Versioned reader profiles; importing this module does not import torch."""

from dataclasses import asdict, dataclass, fields

FP32 = "compatible-fp32"
BF16 = "fixed-voice-bf16"
PROFILES = (FP32, BF16)
BF16_RUNTIME_ABI = "indextts2.5-reader-runtime-v3"


@dataclass(frozen=True)
class InferenceOptimizations:
    compact_mask: bool = True
    shared_gpt_mask: bool = True
    broadcast: bool = True
    dynamic_cache: bool = True
    cpu_beam: bool = True
    incremental_position: bool = True
    euler_invariants: bool = True
    discard_euler_history: bool = True
    inference_mode: bool = True
    voice_cache: bool = True
    release_intermediates: bool = True

    @classmethod
    def baseline(cls):
        return cls(**{field.name: False for field in fields(cls)})

    @classmethod
    def parse(cls, value: str):
        if value == "all":
            return cls()
        if value == "none":
            return cls.baseline()
        enabled = set(value.split(","))
        names = {field.name for field in fields(cls)}
        if enabled - names:
            raise ValueError(f"Unknown optimizations: {sorted(enabled - names)}")
        return cls(**{name: name in enabled for name in names})

    def to_dict(self):
        return asdict(self)


def generation_options(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown reader profile: {profile}")
    options = dict(do_sample=False, num_beams=3, repetition_penalty=10.0,
                   length_penalty=0.0, max_generate_length=1500, num_return_sequences=1)
    if profile == BF16:
        options.update(do_sample=True, top_p=0.8, top_k=30, temperature=0.8)
    return options
