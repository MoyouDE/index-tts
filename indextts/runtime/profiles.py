"""Versioned reader profiles; importing this module does not import torch."""

from dataclasses import asdict, dataclass, fields
import math

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


def synthesis_settings(profile, overrides=None, *, max_tokens=8192):
    """Validate user controls without changing the historical profile defaults."""
    result = {**generation_options(profile), "acoustic_steps": 25, "cfg": 0.7}
    if overrides is None:
        return result
    ranges = {
        "temperature": (0.1, 2.0), "top_p": (0.0, 1.0), "top_k": (0, 100),
        "num_beams": (1, 10), "repetition_penalty": (0.1, 20.0),
        "length_penalty": (-2.0, 2.0), "max_generate_length": (50, max_tokens),
        "acoustic_steps": (1, 100), "cfg": (0.0, 3.0),
    }
    integers = {"top_k", "num_beams", "max_generate_length", "acoustic_steps"}
    if not isinstance(overrides, dict) or set(overrides) - (set(ranges) | {"do_sample"}):
        raise ValueError("generationSettings 含未知参数或不是对象")
    for key, value in overrides.items():
        if key == "do_sample":
            if not isinstance(value, bool):
                raise ValueError("do_sample 必须为布尔值")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} 必须为有限数值")
            low, high = ranges[key]
            if not low <= value <= high or (key in integers and int(value) != value):
                raise ValueError(f"{key} 必须位于 [{low}, {high}]" + (" 且为整数" if key in integers else ""))
            value = int(value) if key in integers else float(value)
        result[key] = value
    return result
