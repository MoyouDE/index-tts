import copy
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.beam_search import BeamSearchScorer

from indextts.gpt.cpu_beam import CpuBeamSearchScorer
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.runtime.engine import ReaderRuntime, validate_seed
from indextts.runtime.profiles import BF16, FP32, InferenceOptimizations, generation_options
from indextts.runtime.sidecar import JsonlSidecar
from indextts.s2mel.modules.flow_matching import BASECFM
from indextts.voicepack.archive import VoicePackError, write_voicepack, load_voicepack
from indextts.voicepack.provenance import PREPROCESS_FINGERPRINT
from test_voicepack import _manifest, _tensors, _licenses


def tiny_gpt(device, dtype=torch.float32):
    torch.manual_seed(5)
    model = UnifiedVoice(layers=2, model_dim=16, heads=2, max_text_tokens=12,
                         max_mel_tokens=32, number_text_tokens=32, number_mel_codes=64,
                         start_mel_token=62, stop_mel_token=63, checkpointing=False,
                         condition_module=None, emo_condition_module=None,
                         spk_cond_mode="campplus", precomputed_conditioning=True)
    model = model.eval().to(device=device, dtype=dtype)
    model.post_init_gpt2_config(kv_cache=True)
    return model


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("optimization", ["dynamic_cache", "cpu_beam", "incremental_position", "all"])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32),
                                          pytest.param("cuda", torch.float32, marks=pytest.mark.gpu),
                                          pytest.param("cuda", torch.bfloat16, marks=pytest.mark.gpu)])
def test_generation_matches_legacy(sample, optimization, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = tiny_gpt(device, dtype)
    condition = torch.ones(1, 3, 16, device=device, dtype=dtype)
    text = torch.tensor([[2, 3, 4, 1]], device=device)
    seen = []
    handle = model.gpt.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(type(kwargs.get("past_key_values"))), with_kwargs=True)

    def generate():
        torch.manual_seed(7)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            return model.inference_speech_from_conditioning(
                condition, text, torch.tensor([0], device=device), max_generate_length=12,
                do_sample=sample, num_beams=3, return_dict_in_generate=True,
                output_scores=True, return_legacy_cache=False)

    before = generate()
    flags = InferenceOptimizations.parse(optimization)
    model.inference_model._supports_cache_class = flags.dynamic_cache
    model.inference_model.reader_incremental_position = flags.incremental_position
    if flags.cpu_beam:
        model.inference_model.beam_scorer_factory = CpuBeamSearchScorer
    seen.clear()
    after = generate()
    handle.remove()
    assert torch.equal(before.sequences, after.sequences)
    assert torch.equal(before.sequences_scores, after.sequences_scores)
    assert all(torch.equal(a, b) for a, b in zip(before.scores, after.scores))
    if flags.dynamic_cache:
        assert seen and all(kind is DynamicCache for kind in seen)
        assert isinstance(after.past_key_values, DynamicCache)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_beam_eos_and_finalize_keep_original_rules(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    inputs = torch.tensor([[1, 2], [1, 3], [1, 4]], device=device)
    scores = torch.tensor([[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]], device=device)
    tokens = torch.tensor([[9, 5, 6, 9, 7, 8]], device=device)
    indices = torch.tensor([[0, 0, 1, 1, 2, 2]], device=device)
    outputs = []
    for cls in (BeamSearchScorer, CpuBeamSearchScorer):
        scorer = cls(batch_size=1, num_beams=3, device=torch.device(device))
        result = scorer.process(inputs, scores, tokens, indices, pad_token_id=0, eos_token_id=9)
        finished = scorer.finalize(inputs, result["next_beam_scores"], tokens, indices,
                                   max_length=4, pad_token_id=0, eos_token_id=9)
        outputs.append((result, finished))
    for a, b in zip(*outputs):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] is b[key] is None or torch.equal(a[key], b[key])


class Estimator(torch.nn.Module):
    def forward(self, x, prompt, lengths, time, style, mu):
        return x * 0.1 + prompt * 0.2 + mu.transpose(1, 2) * 0.3 + style[:, :, None] * 0.1


@pytest.mark.parametrize("cfg", [0.0, 0.7])
@pytest.mark.parametrize("zero_prompt", [False, True])
def test_euler_invariants_and_final_state_are_bit_exact(cfg, zero_prompt):
    solver = BASECFM(SimpleNamespace(DiT=SimpleNamespace(in_channels=2, zero_prompt_speech_token=zero_prompt), reg_loss_type="l1"))
    solver.estimator = Estimator()
    torch.manual_seed(9)
    args = (torch.randn(1, 2, 7), torch.tensor([7]), torch.randn(1, 2, 3),
            torch.randn(1, 7, 2), torch.randn(1, 2), None, torch.linspace(0, 1, 26))
    with torch.inference_mode():
        baseline = solver.solve_euler(*copy.deepcopy(args), inference_cfg_rate=cfg)
        solver.reader_euler_invariants = solver.reader_discard_history = True
        optimized = solver.solve_euler(*copy.deepcopy(args), inference_cfg_rate=cfg)
    assert torch.equal(baseline, optimized)
    assert torch.count_nonzero(optimized[..., :3]) == 0


def test_v2_bf16_pack_requires_producer_precision_and_finite_values(tmp_path):
    manifest = _manifest()
    manifest.update(schemaVersion=2, provenance={
        "profile": BF16, "referenceSha256": "1" * 64,
        "referenceEncoderFingerprint": "2" * 64,
        "preprocessFingerprint": PREPROCESS_FINGERPRINT, "producerVersions": {},
    })
    tensors = _tensors()
    path = write_voicepack(tmp_path / "valid.ivp", manifest, tensors, _licenses())
    assert load_voicepack(path).manifest["provenance"]["profile"] == BF16
    tensors["base_emotion"] = tensors["base_emotion"].float()
    with pytest.raises(VoicePackError, match="precision"):
        write_voicepack(tmp_path / "wrong.ivp", manifest, tensors, _licenses())
    tensors = _tensors()
    tensors["ref_mel"][0, 0, 0] = float("nan")
    with pytest.raises(VoicePackError, match="Non-finite"):
        write_voicepack(tmp_path / "nan.ivp", manifest, tensors, _licenses())


def test_voice_cache_switch_and_queued_snapshot_survive_reload():
    runtime = ReaderRuntime.__new__(ReaderRuntime)
    runtime.device, runtime.dtype = torch.device("cpu"), torch.float32
    runtime.optimizations = InferenceOptimizations()
    runtime._active_pack = runtime._active_tensors = None
    first, second = SimpleNamespace(tensors=_tensors()), SimpleNamespace(tensors=_tensors())
    second.tensors["speaker_latent"] += 1
    runtime._voices = {"first": first, "second": second}
    a = runtime._voice_condition("first", None)[0].clone()
    b = runtime._voice_condition("second", None)[0].clone()
    runtime._voices = {}
    again = runtime._voice_condition("first", None, first)[0]
    assert not torch.equal(a, b)
    assert torch.equal(a, again)
    assert runtime._active_pack is first


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "1"])
def test_seed_rejects_invalid_values(seed):
    with pytest.raises(ValueError):
        validate_seed(seed)


def test_profile_defaults_are_distinct():
    assert generation_options(BF16)["do_sample"] is True
    assert generation_options(FP32)["do_sample"] is False
    assert generation_options(BF16)["top_k"] == 30
    assert not any(InferenceOptimizations.baseline().to_dict().values())


def test_queue_captures_pack_seed_and_mutable_vector():
    pack = object()
    runtime = SimpleNamespace(emotion_provider=None, snapshot_voice=lambda voice: pack)
    server = JsonlSidecar(runtime, stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO())
    params = {"voiceId": "voice", "text": "test", "seed": 4, "emotion": [0.1] * 8}
    server._enqueue_synthesis("1", params)
    params["emotion"][0] = 1.0
    runtime.snapshot_voice = lambda voice: object()
    task = server._queue.get_nowait()
    assert task.params["emotion"][0] == 0.1
    assert task.params["seed"] == 4 and task.voice_pack is pack


@pytest.mark.parametrize("causal", [False, True])
def test_compact_mask_preserves_positions_and_broadcast_output(causal):
    from omegaconf import OmegaConf
    from indextts.s2mel.modules.diffusion_transformer import DiT
    cfg = OmegaConf.create({"style_encoder": {"dim": 4}, "DiT": {
        "time_as_token": False, "style_as_token": False, "uvit_skip_connection": False,
        "depth": 2, "num_heads": 2, "hidden_dim": 16, "in_channels": 2,
        "content_type": "continuous", "content_codebook_size": 8, "content_dim": 4,
        "is_causal": causal, "final_layer_type": "mlp", "style_condition": True,
        "class_dropout_prob": 0, "long_skip_connection": True,
    }})
    torch.manual_seed(11)
    model = DiT(cfg).eval()
    model.setup_caches(1, 64)
    inputs = (torch.randn(2, 2, 41), torch.randn(2, 2, 41), torch.tensor([41]),
              torch.tensor([0.2, 0.2]), torch.randn(2, 4), torch.randn(2, 41, 4))
    with torch.inference_mode():
        before = model(*inputs)
        frequencies = model.transformer.freqs_cis.clone()
        model.reader_compact_mask = model.reader_broadcast = True
        model.transformer.max_seq_length = -1  # rebuild the test instance as runtime does
        model.setup_caches(1, 64)
        after = model(*inputs)
    assert model.transformer.causal_mask.shape == ((64, 64) if causal else (8, 8))
    assert torch.equal(frequencies, model.transformer.freqs_cis)
    assert torch.equal(before, after)


@pytest.mark.gpu
def test_shared_gpt_masks_stay_shared_after_cuda_transfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    runtime = ReaderRuntime.__new__(ReaderRuntime)
    runtime.gpt = tiny_gpt("cpu")
    runtime._share_gpt_masks(torch.device("cuda"))
    runtime.gpt.to("cuda")
    pointers = {layer.attn.bias.data_ptr() for layer in runtime.gpt.gpt.h}
    assert len(pointers) == 1


@pytest.mark.gpu
def test_seed_restores_rng_even_on_failure():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import random
    import numpy as np
    from indextts.runtime.engine import seeded_request
    state = torch.random.get_rng_state()
    gpu_state = torch.cuda.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with pytest.raises(RuntimeError, match="cancelled"):
        with seeded_request(42, torch.device("cuda:0")):
            torch.rand(4, device="cuda")
            torch.rand(4)
            random.random()
            np.random.rand()
            raise RuntimeError("cancelled")
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(gpu_state, torch.cuda.get_rng_state())
    assert python_state == random.getstate()
    actual_numpy = np.random.get_state()
    assert numpy_state[0] == actual_numpy[0]
    assert np.array_equal(numpy_state[1], actual_numpy[1])
    assert numpy_state[2:] == actual_numpy[2:]


def test_benchmark_fixture_has_five_valid_length_groups():
    from pathlib import Path
    cases = json.loads((Path(__file__).parent / "fixtures/reader-benchmark.json").read_text(encoding="utf-8"))
    assert len(cases) == 20
    for case in cases:
        minimum, maximum = map(int, case["group"].split("-"))
        assert minimum <= len(case["text"]) <= maximum


def test_bf16_zero_vector_uses_base_addition_precision():
    runtime = ReaderRuntime.__new__(ReaderRuntime)
    runtime.device, runtime.dtype = torch.device("cpu"), torch.bfloat16
    tensors = _tensors()
    torch.manual_seed(12)
    tensors["speaker_latent"] = torch.randn(1, 1280).bfloat16()
    tensors["base_emotion"] = torch.randn(1, 1280).bfloat16()
    runtime._voices = {"voice": SimpleNamespace(tensors=tensors)}
    base = runtime._voice_condition("voice", None)[0]
    zero = runtime._voice_condition("voice", [0.0] * 8)[0]
    assert base.dtype == zero.dtype == torch.bfloat16
    assert torch.equal(base, zero)


@pytest.mark.parametrize("runtime_profile,pack_profile", [(FP32, BF16), (BF16, FP32)])
def test_runtime_rejects_mixed_explicit_voice_profiles(tmp_path, monkeypatch, runtime_profile, pack_profile):
    (tmp_path / "voice.ivp").touch()
    pack = SimpleNamespace(manifest={"provenance": {"profile": pack_profile}})
    monkeypatch.setattr("indextts.runtime.engine.load_voicepack", lambda *args, **kwargs: pack)
    runtime = ReaderRuntime.__new__(ReaderRuntime)
    runtime.voice_dirs, runtime.profile = [tmp_path], runtime_profile
    runtime.source_fingerprint = "unused"
    with pytest.raises(VoicePackError, match="precision profile"):
        runtime.reload_voices()
