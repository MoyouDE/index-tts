"""MacBERT encoder with Readest's eight-dimensional emotion heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_BASE_MODEL = "hfl/chinese-macbert-base"
MODEL_WEIGHTS = "model.safetensors"


@dataclass
class EmotionModelOutput:
    emotion_logits: torch.Tensor
    intensity_logits: torch.Tensor

    @property
    def intensity(self) -> torch.Tensor:
        return torch.sigmoid(self.intensity_logits)

    @property
    def emotion_vector(self) -> torch.Tensor:
        return torch.sigmoid(self.emotion_logits) * self.intensity


class MacBertEmotionModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, *, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden_size, 8)
        self.intensity_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path = DEFAULT_BASE_MODEL,
        *,
        dropout: float = 0.1,
        local_files_only: bool = False,
    ) -> "MacBertEmotionModel":
        encoder = AutoModel.from_pretrained(
            str(model_name_or_path), local_files_only=local_files_only
        )
        return cls(encoder, int(encoder.config.hidden_size), dropout=dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> EmotionModelOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = self.dropout(encoded.last_hidden_state[:, 0])
        return EmotionModelOutput(
            emotion_logits=self.emotion_head(pooled),
            intensity_logits=self.intensity_head(pooled),
        )


class EmotionOnnxWrapper(nn.Module):
    """Stable ONNX ABI: two FP32 outputs with dynamic batch and sequence axes."""

    def __init__(self, model: MacBertEmotionModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids, attention_mask, token_type_ids)
        return output.emotion_vector.float(), output.intensity.float()


def masked_emotion_loss(
    output: EmotionModelOutput,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    intensity: torch.Tensor,
    *,
    intensity_weight: float = 0.35,
    positive_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    # The eight heads learn composition conditional on active intensity. The
    # exported vector multiplies this distribution by the independent gate so
    # neutral/base examples naturally approach an all-zero explicit vector.
    conditional_labels = torch.where(
        intensity > 1e-6,
        labels / intensity.clamp_min(1e-6),
        torch.zeros_like(labels),
    ).clamp(0.0, 1.0)
    emotion_raw = nn.functional.binary_cross_entropy_with_logits(
        output.emotion_logits,
        conditional_labels,
        reduction="none",
        pos_weight=positive_weights,
    )
    # Neutral/base rows are learned exclusively by the independent intensity
    # gate.  Treating their conditional composition as eight negative labels
    # overwhelms rare emotions and contradicts the conditional-head design.
    active_mask = (intensity > 1e-6).to(label_mask.dtype)
    composition_mask = label_mask * active_mask
    emotion_loss = (emotion_raw * composition_mask).sum() / composition_mask.sum().clamp_min(1.0)
    intensity_loss = nn.functional.binary_cross_entropy_with_logits(
        output.intensity_logits, intensity
    )
    total = emotion_loss + float(intensity_weight) * intensity_loss
    return total, {
        "loss": float(total.detach()),
        "emotionLoss": float(emotion_loss.detach()),
        "intensityLoss": float(intensity_loss.detach()),
    }


def save_checkpoint(model: MacBertEmotionModel, tokenizer, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target)
    model.encoder.config.save_pretrained(target)
    weights = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, target / MODEL_WEIGHTS)
    return target


def load_checkpoint(path: str | Path) -> tuple[MacBertEmotionModel, object]:
    source = Path(path)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    encoder = AutoModel.from_config(config)
    model = MacBertEmotionModel(encoder, int(config.hidden_size))
    missing, unexpected = model.load_state_dict(load_file(source / MODEL_WEIGHTS), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"情感模型权重 ABI 不匹配: missing={missing}, unexpected={unexpected}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    return model, tokenizer
