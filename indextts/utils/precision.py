"""Reload GPT precision without reloading voice conditioning or other TTS models."""

import gc

import torch

from indextts.utils.checkpoint import load_checkpoint


def reload_gpt_precision(tts, half_precision, *, is_v25, use_deepspeed=False):
    """Caller must serialize this operation with inference and voice-pack builds.

    Read original weights on every transition, including BF16/FP16 -> FP32.
    Keep the previous GPT on CPU until the replacement is ready for rollback.
    """
    precision_attr = "use_bf16" if is_v25 else "use_fp16"
    if bool(getattr(tts, precision_attr)) == half_precision:
        return
    if half_precision:
        if not str(tts.device).startswith("cuda"):
            raise ValueError("16-bit switching requires a CUDA device")
        with torch.cuda.device(tts.device):
            if is_v25 and not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 is not supported on this CUDA device")

    # These engines own additional GPU state outside the normal module tree.
    if tts.use_accel or use_deepspeed:
        raise ValueError("Restart with --fp16 to change precision with an acceleration engine")

    target_dtype = (torch.bfloat16 if is_v25 else torch.float16) if half_precision else torch.float32
    old_gpt = tts.gpt
    kwargs = dict(tts.cfg.gpt)
    kwargs["use_accel"] = False
    if is_v25:
        kwargs["spk_cond_mode"] = "campplus"
    new_gpt = type(old_gpt)(**kwargs)
    load_checkpoint(new_gpt, tts.gpt_path)
    new_gpt.eval().to(dtype=target_dtype)
    try:
        old_gpt.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        new_gpt.to(tts.device)
        new_gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=half_precision)
    except Exception:
        # Release any partially uploaded weights before restoring the old model.
        new_gpt.cpu()
        del new_gpt
        gc.collect()
        torch.cuda.empty_cache()
        old_gpt.to(tts.device)
        raise

    tts.gpt = new_gpt
    setattr(tts, precision_attr, half_precision)
    tts.dtype = target_dtype if half_precision else None
    del old_gpt
    gc.collect()
    torch.cuda.empty_cache()
