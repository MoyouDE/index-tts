"""Isolated identity-fusion candidates; never used by the formal pack builder."""


def fuse_identity(primary, styles, weight, bias, method):
    import torch
    from torch.nn import functional as F
    if method not in {"mean", "normalized"} or not styles:
        raise ValueError("未知身份融合方法或空片段集合")
    if any(s.dtype != torch.float32 or s.shape != primary["speaker_style"].shape or not torch.isfinite(s).all() for s in styles):
        raise ValueError("实验身份向量必须是相同形状的有限 FP32 张量")
    stacked=torch.stack(styles)
    if method=="mean":
        style=stacked.mean(0)
    else:
        norms=stacked.norm(dim=-1,keepdim=True)
        if (norms<=1e-8).any():
            raise ValueError("身份向量范数为零")
        style=(stacked/norms).mean(0)*norms.mean(0)
    result={name:tensor.clone() for name,tensor in primary.items()}
    result["speaker_style"]=style.contiguous()
    result["speaker_latent"]=F.linear(style,weight,bias).contiguous()
    return result
