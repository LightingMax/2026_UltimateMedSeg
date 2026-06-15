"""SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
    SegFormer: 简单 and 高效的 Design for 语义的 分割 with Transformers。

中文: SegFormer：基于 Transformer 的简洁高效语义分割。

Reference:
    Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers", NeurIPS 2021.
    Source: https://github.com/NVlabs/SegFormer

Architecture overview:
  - Encoder: MiT (Mix-Transformer) backbone from timm, sizes B0-B5
  - Decoder: All-MLP decoder (project all multi-scale features to a common
    embed_dim, upsample, concat, fuse with 1x1 conv + BN + GELU)
  - Head: 1x1 conv segmentation head

Six model sizes are registered:
  segformer_b0  MiT-B0  [32, 64, 160, 256]    layers=[2,2,2,2]
  segformer_b1  MiT-B1  [64, 128, 320, 512]   layers=[2,2,2,2]
  segformer_b2  MiT-B2  [64, 128, 320, 512]   layers=[3,4,6,3]
  segformer_b3  MiT-B3  [64, 128, 320, 512]   layers=[3,4,18,3]
  segformer_b4  MiT-B4  [64, 128, 320, 512]   layers=[3,8,27,3]
  segformer_b5  MiT-B5  [64, 128, 320, 512]   layers=[3,8,27,3]
"""
# Source: https://github.com/NVlabs/SegFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import timm


# ---------------------------------------------------------------------------
# MiT 骨干网络 helpers / MiT backbone helpers
# ---------------------------------------------------------------------------

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback as _ssl_load


def _create_mit(model_name: str, pretrained: bool, in_chans: int = 3):
    """Create a MiT 骨干网络 via timm with PVTv2 fallback for older versions。
        Create a MiT backbone via timm with PVTv2 fallback for older versions."""
    # MiT candidates and their PVTv2 aliases
    _FALLBACK = {
        'mit_b0': 'pvt_v2_b0',
        'mit_b1': 'pvt_v2_b1',
        'mit_b2': 'pvt_v2_b2',
        'mit_b3': 'pvt_v2_b3',
        'mit_b4': 'pvt_v2_b4',
        'mit_b5': 'pvt_v2_b5',
    }
    candidates = [model_name]
    if model_name in _FALLBACK:
        candidates.append(_FALLBACK[model_name])

    last_err = None
    for name in candidates:
        try:
            def _create(pretrained_flag, _name=name):
                return timm.create_model(
                    _name, pretrained=pretrained_flag,
                    features_only=True, in_chans=in_chans,
                )
            return _ssl_load(_create, pretrained)
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not construct MiT backbone '{model_name}' "
        f"(also tried {candidates[1:]}): {last_err}"
    )


# ---------------------------------------------------------------------------
# SegFormer All-MLP 解码器 / SegFormer All-MLP Decoder (inline, faithful to original)
# ---------------------------------------------------------------------------

class _MLPDecoder(nn.Module):
    """SegFormer all-MLP 解码器。
        SegFormer all-MLP decoder: project, upsample, concat, fuse."""

    def __init__(self, encoder_channels: List[int], embed_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 1x1 conv projection per scale → common embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, 1, bias=False)
            for ch in encoder_channels
        ])

        # 融合: concat → 1x1 conv → BN → GELU / Fusion: concat → 1x1 conv → BN → GELU
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(encoder_channels), embed_dim,
                      1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 特征: shallow-first, deep-last / features: shallow-first, deep-last
        target_size = features[0].shape[2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            p = proj(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size,
                                  mode='bilinear', align_corners=False)
            projected.append(p)
        x = self.fuse(torch.cat(projected, dim=1))
        return self.dropout(x)


# ---------------------------------------------------------------------------
# SegFormer
# ---------------------------------------------------------------------------

class SegFormer(nn.Module):
    """SegFormer (NeurIPS 2021): MiT encoder + All-MLP decoder + seg head.
        SegFormer (NeurIPS 2021): MiT 编码器。

    中文: SegFormer（NeurIPS 2021）：MiT 编码器 + 全 MLP 解码器 + 分割头。

    Six sizes available (B0-B5), controlled by ``backbone`` parameter.

    Args:
        in_channels: Input channels (default 3, RGB expected by MiT).
        num_classes: Output segmentation classes (default 2).
        img_size: Input spatial size (default 224).
        backbone: MiT backbone size, one of
            ``'mit_b0'`` .. ``'mit_b5'`` (default ``'mit_b2'``).
        embed_dim: Decoder embedding dimension (default 256 for B2-B5,
            128 for B0).
        pretrained: Load ImageNet-pretrained MiT weights.
    """

    # 默认值 embed _ dim per 骨干网络 大小 / Default embed_dim per backbone size
    _DEFAULT_EMBED = {
        'mit_b0': 128, 'mit_b1': 256, 'mit_b2': 256,
        'mit_b3': 256, 'mit_b4': 256, 'mit_b5': 256,
    }

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 img_size: int = 224, backbone: str = 'mit_b2',
                 embed_dim: int = None, pretrained: bool = True,
                 deep_supervision: bool = False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 通道 remap for non-RGB inputs / Channel remap for non-RGB inputs
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, 1, bias=False)
        else:
            self.input_proj = None

        # MiT 编码器 / MiT encoder backbone
        if pretrained:
            self.encoder = _ssl_load(
                _create_mit, backbone, True, 3)
        else:
            self.encoder = _create_mit(backbone, False, 3)

        enc_channels = list(self.encoder.feature_info.channels())
        # Keep deepest 4 特征 ( some timm versions may emit 5 ) / Keep deepest 4 features (some timm versions may emit 5)
        if len(enc_channels) > 4:
            self._enc_slice = slice(len(enc_channels) - 4, len(enc_channels))
            enc_channels = enc_channels[self._enc_slice]
        else:
            self._enc_slice = slice(0, len(enc_channels))
        self._enc_channels = enc_channels  # for BCHW detection in forward

        # 解码 / Decoder
        if embed_dim is None:
            embed_dim = self._DEFAULT_EMBED.get(backbone, 256)
        self.decoder = _MLPDecoder(enc_channels, embed_dim=embed_dim)

        # 分割 头部 / Segmentation head
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        if self.input_proj is not None:
            x = self.input_proj(x)

        # 编码器: 提取 多尺度 特征 / Encoder: extract multi-scale features
        feats = list(self.encoder(x))[self._enc_slice]
        # Ensure BCHW ( some timm versions 输出 BHWC ) / Ensure BCHW (some timm versions output BHWC).
        # Use known 编码器 / Use known encoder channels for reliable format detection.
        bchw_feats = []
        for i, f in enumerate(feats):
            if f.ndim == 4 and f.shape[1] != self._enc_channels[i]:
                f = f.permute(0, 3, 1, 2).contiguous()
            bchw_feats.append(f)

        # 解码 / Decoder
        d = self.decoder(bchw_feats)

        # 头部 / Head
        out = self.head(d)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size,
                                mode='bilinear', align_corners=False)
        return out


__all__ = ['SegFormer']
