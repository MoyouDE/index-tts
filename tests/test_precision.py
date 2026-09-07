from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from indextts.utils import precision


class TinyGPT(torch.nn.Module):
    fail_init = False

    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0012345]))

    def to(self, *args, **kwargs):
        # Exercise actual dtype conversions without allocating CUDA memory.
        if args and str(args[0]).startswith("cuda"):
            return self
        return super().to(*args, **kwargs)

    def post_init_gpt2_config(self, **kwargs):
        if self.fail_init:
            raise RuntimeError("simulated initialization failure")


@pytest.fixture
def tts(tmp_path, monkeypatch):
    model = TinyGPT()
    checkpoint = tmp_path / "gpt.pth"
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return SimpleNamespace(
        gpt=model, gpt_path=str(checkpoint), cfg=SimpleNamespace(gpt={}),
        device="cuda:0", use_accel=False, use_bf16=False, use_fp16=False, dtype=None,
    )


@pytest.mark.parametrize("is_v25,dtype,flag", [
    (True, torch.bfloat16, "use_bf16"),
    (False, torch.float16, "use_fp16"),
])
def test_round_trip_restores_original_fp32_weights(tts, is_v25, dtype, flag):
    original = tts.gpt.weight.detach().clone()
    precision.reload_gpt_precision(tts, True, is_v25=is_v25)
    assert tts.gpt.weight.dtype == dtype
    assert tts.dtype == dtype and getattr(tts, flag)
    assert not torch.equal(tts.gpt.weight.float(), original)

    precision.reload_gpt_precision(tts, False, is_v25=is_v25)
    assert tts.gpt.weight.dtype == torch.float32
    assert tts.dtype is None and not getattr(tts, flag)
    assert torch.equal(tts.gpt.weight, original)


def test_failed_switch_preserves_model_and_precision(tts, monkeypatch):
    previous = tts.gpt
    monkeypatch.setattr(TinyGPT, "fail_init", True)
    with pytest.raises(RuntimeError, match="simulated"):
        precision.reload_gpt_precision(tts, True, is_v25=True)
    assert tts.gpt is previous
    assert tts.gpt.weight.dtype == torch.float32
    assert not tts.use_bf16 and tts.dtype is None


def test_unsupported_bf16_does_not_mutate_model(tts, monkeypatch):
    previous = tts.gpt
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="BF16"):
        precision.reload_gpt_precision(tts, True, is_v25=True)
    assert tts.gpt is previous
    assert not tts.use_bf16


def test_unchanged_precision_does_not_reload_weights(tts, monkeypatch):
    def unexpected_load(*args):
        pytest.fail("unchanged precision should not reload")

    monkeypatch.setattr(precision, "load_checkpoint", unexpected_load)
    precision.reload_gpt_precision(tts, False, is_v25=True)
