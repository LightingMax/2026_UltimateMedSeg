"""CCNet Decoder — Criss-Cross Attention for Semantic Segmentation.
    CCNet 解码器。

中文: CCNet 解码器：十字交叉注意力语义分割。

Reference:
    Huang et al., "CCNet: Criss-Cross Attention for Semantic Segmentation",
    ICCV 2019.
    Source: https://github.com/speedinghzl/CCNet
    Pure-PyTorch: https://github.com/Serge-weihao/CCNet-Pure-Pytorch

Architecture:
    1. Apply 1×1 conv to reduce bottleneck channels → ``mid_ch``.
    2. Run **Criss-Cross Attention** (RCCA) twice in a recurrent loop:
       - For every pixel (i, j), compute attention along its row *and*
         column only (H + W − 1 positions instead of H × W), dramatically
         cutting cost vs. full self-attention.
       - After two passes the receptive field covers the entire spatial
         extent: each pixel has indirectly attended every other pixel.
    3. Fuse the attended feature with the original via residual + 1×1 conv.

    ``has_internal_skip = True``: CCNet operates purely on the deepest
    feature; external skip connections are IGNORED.

Integration:
    Set ``decoder: { name: ccnet }`` in YAML.
"""
# Source: https://github.com/speedinghzl/CCNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _CrissCrossAttention(nn.Module):
    """Criss-Cross Attention module.
        Criss-Cross 注意力 模块。

    For each pixel at position (i, j), attention is computed only along
    the i-th row and j-th column, giving (H + W − 1) keys per position
    instead of H × W.
    """

    def __init__(self, in_ch: int, mid_ch: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_ch, mid_ch, 1)
        self.key_conv = nn.Conv2d(in_ch, mid_ch, 1)
        self.value_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        Q = self.query_conv(x)  # (B, mid_ch, H, W)
        K = self.key_conv(x)    # (B, mid_ch, H, W)
        V = self.value_conv(x)  # (B, C, H, W)

        # ---- Horizontal affinity: each pixel ↔ entire row ----
        # Q_h: (B, H, W, mid_ch)   K_h: (B, H, W, mid_ch)
        Q_h = Q.permute(0, 2, 3, 1)   # (B, H, W, mid_ch)
        K_h = K.permute(0, 2, 3, 1)   # (B, H, W, mid_ch)
        # energy_h[b, h, w1, w2] = sum_c Q[b,h,w1,c] * K[b,h,w2,c]
        energy_h = torch.matmul(Q_h, K_h.transpose(-1, -2))  # (B, H, W, W)

        # ---- Vertical affinity: each pixel ↔ entire column ----
        # Q_v: (B, W, H, mid_ch)   K_v: (B, W, H, mid_ch)
        Q_v = Q.permute(0, 3, 2, 1)   # (B, W, H, mid_ch)
        K_v = K.permute(0, 3, 2, 1)   # (B, W, H, mid_ch)
        # energy_v[b, w, h1, h2] = sum_c Q[b,w,h1,c] * K[b,w,h2,c]
        energy_v = torch.matmul(Q_v, K_v.transpose(-1, -2))  # (B, W, H, H)

        # - - - - 组合 and softmax over ( H + W - 1 ) keys - - - - / ---- Combine and softmax over (H + W - 1) keys ----
        # We cannot simply softmax each direction separately; we need a
        # joint normalisation over the union of row + column positions.
        #
        # The original CUDA RCCA removes the self-position (i,j) from the
        # horizontal 分支 to avoid double-counting it in the vertical / horizontal branch to avoid double-counting it in the vertical
        # 分支. We replicate this by masking the diagonal ( w1 = = w2 ) of / branch.  We replicate this by masking the diagonal (w1==w2) of
        # the horizontal energy to -inf before the joint softmax.
        diag_mask = torch.eye(W, device=energy_h.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
        energy_h = energy_h.masked_fill(diag_mask, float('-inf'))

        energy_h_flat = energy_h.reshape(B * H * W, W)          # (BHW, W)
        energy_v_flat = energy_v.permute(0, 2, 1, 3)            # (B, H, W, H)
        energy_v_flat = energy_v_flat.reshape(B * H * W, H)     # (BHW, H)

        # 拼接: ( BHW, W + H ) / Concatenate: (BHW, W+H)
        energy_cat = torch.cat([energy_h_flat, energy_v_flat], dim=1)
        energy_cat = F.softmax(energy_cat, dim=1)

        # Split back
        att_h_flat = energy_cat[:, :W]           # (BHW, W)
        att_v_flat = energy_cat[:, W:]           # (BHW, H)
        att_h = att_h_flat.reshape(B, H, W, W)   # (B, H, W, W)
        att_v = att_v_flat.reshape(B, H, W, H)   # (B, H, W, H)

        # - - - - 聚合 values - - - - / ---- Aggregate values ----
        # Horizontal: V_h(b, h, w2, c) × att_h(b, h, w1, w2) → out_h(b, h, w1, c)
        V_h = V.permute(0, 2, 3, 1)             # (B, H, W, C)
        out_h = torch.matmul(att_h, V_h)         # (B, H, W, C)

        # Vertical: V_v(b, w, h2, c) × att_v_reshaped
        # att _ v is ( B, H, W, H ); we need att _ v used as 权重 over rows / att_v is (B, H, W, H); we need att_v used as weights over rows
        # att_v[b, h1, w, h2] × V[b, h2, w, c]
        V_v = V.permute(0, 3, 2, 1)             # (B, W, H, C)
        V_v = V_v.permute(0, 2, 1, 3)           # (B, H, W, C)
        # Now V_v[b, h2, w, c]; att_v[b, h1, w, h2]
        # We need: out_v[b, h1, w, c] = sum_{h2} att_v[b, h1, w, h2] × V_v[b, h2, w, c]
        # att_v: (B, H, W, H)  V_v: (B, H, W, C)
        # 转置 att _ v last two dims for matmul: att _ v [ b, h1, w, h2 ] × V _ v [ b, h2, w, c ] / Transpose att_v last two dims for matmul: att_v[b,h1,w,h2] × V_v[b,h2,w,c]
        # But V_v's "w" dim is shared; we need grouped matmul over H.
        # 重塑: att _ v → ( B * W, H, H ) and V _ v → ( B * W, H, C ) / Reshape: att_v → (B*W, H, H) and V_v → (B*W, H, C)
        att_v_g = att_v.permute(0, 2, 1, 3).reshape(B * W, H, H)  # (BW, H, H)
        V_v_g = V_v.permute(0, 2, 1, 3).reshape(B * W, H, C)      # (BW, H, C)
        out_v_g = torch.bmm(att_v_g, V_v_g)                        # (BW, H, C)
        out_v = out_v_g.reshape(B, W, H, C).permute(0, 2, 1, 3)   # (B, H, W, C)

        out = out_h + out_v                                        # (B, H, W, C)
        out = out.permute(0, 3, 1, 2)                              # (B, C, H, W)
        out = self.gamma * out + x
        return out


@DECODER_REGISTRY.register("ccnet")
class CCNetDecoder(nn.Module):
    """CCNet decoder: recurrent Criss-Cross Attention × ``num_recurrence``.
        CCNet 解码器。

    中文: CCNet 解码器：循环十字交叉注意力。

    Args:
        encoder_channels: list of encoder output channels (unused, kept for API).
        bottleneck_channels: bottleneck output channel count.
        skip_connection: unused (has_internal_skip = True).
        mid_channels: intermediate channel count for Q/K projections
                      (default: bottleneck_channels // 8, min 64).
        num_recurrence: number of times to run Criss-Cross Attention (default 2).
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, mid_channels: int = 0,
                 num_recurrence: int = 2, **kwargs):
        super().__init__()
        in_ch = bottleneck_channels
        if mid_channels <= 0:
            mid_channels = max(64, in_ch // 8)

        # 通道 reduction before RCCA / Channel reduction before RCCA
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        # 循环的 Criss-Cross 注意力 ( shared 权重 ) / Recurrent Criss-Cross Attention (shared weights)
        self.rcca = _CrissCrossAttention(in_ch=in_ch, mid_ch=mid_channels)
        self.num_recurrence = num_recurrence

        # 输出 projection / Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_ch

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]):
        """Forward pass.
            前向传播 pass。

        Args:
            bottleneck_feat: deepest feature after bottleneck (B, C, H, W).
            skip_features: list of skip features (IGNORED by CCNet decoder).
        Returns:
            Context-enriched feature map (B, C, H, W).
        """
        x = self.reduce(bottleneck_feat)
        for _ in range(self.num_recurrence):
            x = self.rcca(x)
        x = self.out_proj(x)
        return x


__all__ = ['CCNetDecoder']
