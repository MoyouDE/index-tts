"""Instance-selectable adapter around the pinned Transformers scoring rules."""

import torch
from transformers.generation.beam_search import BeamSearchScorer


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


class CpuBeamSearchScorer(BeamSearchScorer):
    """Move whole candidate arrays, never individual CUDA scalars, to the scorer."""

    def __init__(self, *args, device, **kwargs):
        super().__init__(*args, device=torch.device("cpu"), **kwargs)

    def process(self, input_ids, next_scores, next_tokens, next_indices, **kwargs):
        device = input_ids.device
        result = super().process(
            input_ids.cpu(), next_scores.cpu(), next_tokens.cpu(), next_indices.cpu(),
            **{key: _move(value, "cpu") for key, value in kwargs.items()},
        )
        return {key: _move(value, device) for key, value in result.items()}

    def finalize(self, input_ids, final_beam_scores, final_beam_tokens, final_beam_indices,
                 max_length, **kwargs):
        device = input_ids.device
        result = super().finalize(
            input_ids.cpu(), final_beam_scores.cpu(), final_beam_tokens.cpu(),
            final_beam_indices.cpu(), max_length,
            **{key: _move(value, "cpu") for key, value in kwargs.items()},
        )
        return {key: _move(value, device) for key, value in result.items()}
