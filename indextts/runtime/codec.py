"""Decoder-only semantic codec used by the reader runtime."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from indextts.codec.amphion_codec.quantize import ResidualVQ
from indextts.codec.kmeans.vocos import VocosBackbone


class SemanticCodecDecoder(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=2,
        cfg=None,
    ):
        super().__init__()
        value = lambda name, default: getattr(cfg, name) if cfg is not None and hasattr(cfg, name) else default
        self.downsample_scale = value("downsample_scale", downsample_scale)
        hidden_size = value("hidden_size", hidden_size)
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=value("vocos_dim", vocos_dim),
                intermediate_dim=value("vocos_intermediate_dim", vocos_intermediate_dim),
                num_layers=value("vocos_num_layers", vocos_num_layers),
                adanorm_num_embeddings=None,
            ),
            nn.Linear(value("vocos_dim", vocos_dim), hidden_size),
        )
        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=value("num_quantizers", num_quantizers),
            codebook_size=value("codebook_size", codebook_size),
            codebook_dim=value("codebook_dim", codebook_dim),
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )
        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.up = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        quantized = self.quantizer.vq2emb(codes)
        decoded = self.decoder(quantized)
        if self.downsample_scale is not None and self.downsample_scale > 1:
            decoded = decoded.transpose(1, 2)
            decoded = F.interpolate(decoded, scale_factor=2, mode="nearest")
            decoded = self.up(decoded).transpose(1, 2)
        return decoded
