"""FF-Parser Decoder — Feature Frequency Parser for Segmentation.
    FF-Parser 解码器。

中文: FF-Parser 解码器：特征频率解析器分割解码器。

Reference:
    Gu et al., "TransDiffSeg: Transformer-Based Conditional Diffusion
    Segmentation Model for Abdominal Multi-Objective", J Imaging Inform
    Med 2024.
    Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11810859/

Architecture:
    1. **Frequency-domain filtering**: Apply 2-D FFT on the bottleneck
       feature, multiply by a learnable low-pass filter mask to suppress
       high-frequency noise, then inverse FFT back to spatial domain.
    2. **Multi-scale feature fusion**: Upsample skip features to match
       the filtered feature resolution, concatenate, and apply a 1x1
       conv to fuse.
    3. **Attentive refinement**: Apply a lightweight channel-spatial
       attention module to the fused feature to re-emphasise informative
       channels and spatial locations.

    ``has_internal_skip = False``: FF-Parser uses skip connections for
    multi-scale feature fusion.

Integration:
    Set ``decoder: { name: ffparser }`` in YAML.
"""
# Source: TransDiffSeg / Gu et al., 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _FrequencyFilter(nn.Module):
    """Learnable frequency-domain low-pass filter.

    Applies 2-D FFT, multiplies by a learnable filter mask in the
    frequency domain, then inverse FFT back to spatial domain.

    The filter mask is initialised as a Gaussian low-pass so that
    high-frequency noise is attenuated from the start.
    """

    def __init__(self, channels: int, spatial_size: int = 64,
                 cutoff_ratio: float = 0.5):
        super().__init__()
        self.channels = channels
        mask_h, mask_w = spatial_size, spatial_size
        self.mask_real = nn.Parameter(torch.ones(1, 1, mask_h, mask_w))
        self.mask_imag = nn.Parameter(torch.zeros(1, 1, mask_h, mask_w))
        self._init_gaussian_lp(mask_h, mask_w, cutoff_ratio)

    def _init_gaussian_lp(self, H: int, W: int, cutoff: float):
        """Initialise the 掩码 as a Gaussian low-pass filter。
            Initialise the mask as a Gaussian low-pass filter."""
        cy, cx = H // 2, W // 2
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist_sq = yy ** 2 + xx ** 2
        sigma_sq = (min(H, W) * cutoff) ** 2
        gauss = torch.exp(-dist_sq / (2 * sigma_sq))
        with torch.no_grad():
            self.mask_real.copy_(gauss.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')

        mask_real = self.mask_real
        mask_imag = self.mask_imag
        if mask_real.shape[2:] != (H, W):
            mask_real = F.interpolate(
                mask_real, size=(H, W), mode='bilinear', align_corners=False)
            mask_imag = F.interpolate(
                mask_imag, size=(H, W), mode='bilinear', align_corners=False)

        mask = torch.complex(mask_real, mask_imag)
        mask = mask.expand(B, C, H, W)
        x_fft_filtered = x_fft * mask
        x_filtered = torch.fft.ifft2(x_fft_filtered, norm='ortho').real
        return x_filtered


class _ChannelSpatialAttention(nn.Module):
    """轻量级 通道 + 空间的 注意力 refinement 模块。
        Lightweight channel + spatial attention refinement module."""

    def __init__(self, in_ch: int, reduction: int = 8):
        super().__init__()
        mid_ch = max(in_ch // reduction, 16)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_attn(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


@DECODER_REGISTRY.register("ffparser")
class FFParserDecoder(nn.Module):
    """FF-Parser decoder: frequency-domain feature filtering + fusion.
        FF-Parser 解码器。

    中文: FF-Parser 解码器：频域特征过滤与融合。

    Args:
        encoder_channels: list of encoder output channel counts.
        bottleneck_channels: bottleneck output channels.
        skip_connection: unused directly (uses skip_features internally).
        hidden_channels: intermediate / output channel count (default 256).
        cutoff_ratio: initial low-pass filter cutoff ratio (default 0.5).
    """
    has_internal_skip = False

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, hidden_channels: int = 256,
                 cutoff_ratio: float = 0.5, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(bottleneck_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.freq_filter = _FrequencyFilter(
            channels=hidden_channels, spatial_size=64,
            cutoff_ratio=cutoff_ratio,
        )

        self._skip_projs = None
        self._n_skip = None
        self._fusion_conv = None

        self.refine = _ChannelSpatialAttention(hidden_channels)
        self._out_channels = hidden_channels

    @property
    def out_channels(self):
        return self._out_channels

    def _build_skip_projs(self, skip_features: List[torch.Tensor],
                           device: torch.device):
        """Lazily build 跳跃连接。
            Lazily build skip projection layers on first forward call."""
        self._n_skip = len(skip_features)
        self._skip_projs = nn.ModuleList()
        for sf in skip_features:
            ch = sf.shape[1]
            proj = nn.Sequential(
                nn.Conv2d(ch, self.hidden_channels, 1, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(inplace=True),
            ).to(device)
            self._skip_projs.append(proj)
        total_in = self.hidden_channels * (1 + self._n_skip)
        self._fusion_conv = nn.Sequential(
            nn.Conv2d(total_in, self.hidden_channels, 1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
        ).to(device)

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        if self._skip_projs is None:
            self._build_skip_projs(skip_features, bottleneck_feat.device)

        bn_feat = self.bottleneck_proj(bottleneck_feat)
        bn_filtered = self.freq_filter(bn_feat)

        target_size = skip_features[0].shape[2:]
        if bn_filtered.shape[2:] != target_size:
            bn_filtered = F.interpolate(
                bn_filtered, size=target_size,
                mode='bilinear', align_corners=False)

        skip_projected = []
        for sf, proj in zip(skip_features, self._skip_projs):
            sp = proj(sf)
            if sp.shape[2:] != target_size:
                sp = F.interpolate(
                    sp, size=target_size,
                    mode='bilinear', align_corners=False)
            skip_projected.append(sp)

        fused = torch.cat([bn_filtered] + skip_projected, dim=1)
        fused = self._fusion_conv(fused)
        out = self.refine(fused)
        return out


__all__ = ['FFParserDecoder']
