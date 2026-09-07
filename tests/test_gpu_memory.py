from contextlib import nullcontext

import pytest
import torch

from indextts.utils import gpu_memory


class TrackingModel:
    def __init__(self, fail_transfer=False):
        self.device = "cpu"
        self.fail_transfer = fail_transfer

    def to(self, device):
        self.device = device
        if self.fail_transfer:
            raise RuntimeError("partial upload")
        return self

    def cpu(self):
        self.device = "cpu"
        return self


@pytest.fixture
def releases(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("released"))
    return calls


def test_auxiliary_model_leaves_gpu_after_success(releases):
    model = TrackingModel()
    with gpu_memory.staged_model(model, "cuda:0") as active:
        assert active is model and model.device == "cuda:0"
    assert model.device == "cpu"
    assert releases == ["released"]


def test_auxiliary_model_leaves_gpu_after_failed_forward(releases):
    model = TrackingModel()
    with pytest.raises(RuntimeError, match="forward"):
        with gpu_memory.staged_model(model, "cuda:0"):
            raise RuntimeError("forward failed")
    assert model.device == "cpu"
    assert releases == ["released"]


def test_partial_upload_is_released(releases):
    model = TrackingModel(fail_transfer=True)
    with pytest.raises(RuntimeError, match="partial upload"):
        with gpu_memory.staged_model(model, "cuda:0"):
            pytest.fail("upload failed")
    assert model.device == "cpu"
    assert releases == ["released"]


def test_cpu_requests_do_not_initialize_cuda(monkeypatch):
    def unexpected(*args):
        pytest.fail("CPU path must not call CUDA")

    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    model = torch.nn.Linear(2, 2)
    original = model(torch.ones(1, 2))
    with gpu_memory.staged_model(model, "cpu"):
        assert torch.equal(model(torch.ones(1, 2)), original)
    gpu_memory.synchronize_device("cpu")
    gpu_memory.release_cuda_cache("cpu")
    assert gpu_memory.cuda_memory_snapshot("cpu") == {}
