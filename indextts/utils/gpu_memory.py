"""Bound GPU residency to the models and requests that actually need it."""

from contextlib import contextmanager

import torch


def synchronize_device(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def release_cuda_cache(device):
    """Release unused allocator blocks at a serialized request boundary.

    On Windows, a large idle allocator cache competes with display applications
    for WDDM's memory budget and can cause subsequent kernels to page to RAM.
    Live model weights and tensors are unaffected by empty_cache().
    """
    if str(device).startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def cuda_memory_snapshot(device):
    if not str(device).startswith("cuda"):
        return {}
    mib = 1024 ** 2
    return {
        "allocatedMiB": round(torch.cuda.memory_allocated(device) / mib, 1),
        "reservedMiB": round(torch.cuda.memory_reserved(device) / mib, 1),
        "peakAllocatedMiB": round(torch.cuda.max_memory_allocated(device) / mib, 1),
    }


@contextmanager
def staged_model(model, device):
    """Auxiliary models live on CPU and visit CUDA only for their own stage.

    Callers serialize model use; outputs may remain on CUDA for the next stage.
    Always return weights to CPU, including when a forward pass fails.
    """
    try:
        model.to(device)
        yield model
    finally:
        if str(device).startswith("cuda"):
            model.cpu()
            release_cuda_cache(device)
