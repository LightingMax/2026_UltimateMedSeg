"""OCRNet Decoder — Object-Contextual Representations for Semantic Segmentation.
    OCRNet 解码器。

中文: OCRNet 解码器：面向语义分割的目标上下文表示。

Reference:
    Yuan et al., "Object-Contextual Representations for Semantic
    Segmentation", ECCV 2020.
    Source: https://github.com/HRNet/HRNet-Semantic-Segmentation
    Unofficial: https://github.com/rosinality/ocr-pytorch

Architecture:
    1. Apply a 1×1 conv on the bottleneck feature to produce a *coarse*
       segmentation prediction (``num_classes`` channels).
    2. **SpatialGather**: Use the coarse prediction (softmax over pixels) as
       soft pixel→region associations.  Aggregate pixel features weighted by
       these associations to produce *object-region* representations (one per
       class).
    3. **SpatialOCR**: Compute affinities between each pixel and every object
       region, normalise with softmax, and aggregate region features to form
       an *OCR-augmented* feature map.
    4. Fuse the original bottleneck feature with the OCR-augmented feature
       via a 1×1 conv + BN + ReLU, then upsample to full resolution.

    ``has_internal_skip = True``: the OCR decoder operates purely on the
    deepest feature; external skip connections are IGNORED.

Integration:
    Set ``decoder: { name: ocrnet }`` in YAML.
"""
# Source: https://github.com/HRNet/HRNet-Semantic-Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import DECODER_REGISTRY


class _SpatialGather(nn.Module):
    """Compute object-region representations from pixel features + soft mask.
        计算 object-region representations from pixel 特征 + soft 掩码。

    Given pixel features ``F`` (B×C×H×W) and a coarse prediction ``probs``
    (B×K×H×W, softmax), produces object-region features ``O`` (B×K×C).

    ``O_k = sum_i (probs_ki * F_i) / sum_i (probs_ki)``
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, pixel_feats: torch.Tensor,
                probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_feats: (B, C, H, W) pixel-level feature maps.
            probs: (B, K, H, W) soft class probabilities (after softmax).
        Returns:
            (B, K, C) object-region representations.
        """
        b, c, h, w = pixel_feats.shape
        k = probs.shape[1]
        # 重塑: pixel _ feats → ( B, C, HW ), probs → ( B, K, HW ) / Reshape: pixel_feats → (B, C, HW), probs → (B, K, HW)
        feats = pixel_feats.reshape(b, c, h * w)           # (B, C, HW)
        probs = probs.reshape(b, k, h * w) * self.scale    # (B, K, HW)
        probs = F.softmax(probs, dim=2)                    # normalise over pixels
        # 聚合: O = probs × feats ^ T → ( B, K, C ) / Aggregate: O = probs × feats^T  →  (B, K, C)
        context = torch.bmm(probs, feats.permute(0, 2, 1))
        return context  # (B, K, C)


class _SpatialOCR(nn.Module):
    """Object-Contextual Representation module.
        Object-Contextual Representation 模块。

    Computes pixel↔object-region affinities and produces an OCR-augmented
    feature map.

    Flow:
        1. Pixel query  = W_pixel(pixel_feats)   → (B, key_ch, H, W)
        2. Region key   = W_region(context)       → (B, K, key_ch)
        3. Affinities   = softmax(pixel_q^T · region_k) over K classes
                                                    → (B, K, H, W)
        4. Region value = W_value(context)        → (B, K, out_ch)
        5. Aggregate    = sum_k affinities_k · region_v_k
                                                    → (B, out_ch, H, W)
        6. Project      = W_project(aggregate)    → (B, out_ch, H, W)
    """

    def __init__(self, in_ch: int, key_ch: int = 256, out_ch: int = 512,
                 num_classes: int = 19, dropout: float = 0.05):
        super().__init__()
        self.num_classes = num_classes
        # Pixel query transform
        self.W_pixel = nn.Sequential(
            nn.Conv2d(in_ch, key_ch, 1, bias=False),
            nn.BatchNorm2d(key_ch),
            nn.ReLU(inplace=True),
        )
        # 区域 key transform ( B, K, C ) → ( B, K, key _ ch ) / Region key transform  (B, K, C) → (B, K, key_ch)
        self.W_region = nn.Linear(in_ch, key_ch)
        # 区域 value transform / Region value transform
        self.W_value = nn.Linear(in_ch, out_ch)
        # Post-aggregation projection
        self.W_project = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, pixel_feats: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_feats: (B, C, H, W)
            context: (B, K, C) object-region representations from SpatialGather
        Returns:
            (B, out_ch, H, W) OCR-augmented features.
        """
        b, c, h, w = pixel_feats.shape
        k = context.shape[1]
        # Pixel query
        pixel_q = self.W_pixel(pixel_feats)                    # (B, key_ch, H, W)
        pixel_q = pixel_q.reshape(b, -1, h * w)                # (B, key_ch, HW)
        # 区域 key / Region key
        region_k = self.W_region(context)                      # (B, K, key_ch)
        # Affinities: (B, HW, key_ch) × (B, key_ch, K) → (B, HW, K)
        affinities = torch.bmm(pixel_q.permute(0, 2, 1), region_k.permute(0, 2, 1))
        affinities = affinities.reshape(b, h * w, k).permute(0, 2, 1)  # (B, K, HW)
        affinities = F.softmax(affinities, dim=1)              # softmax over K classes
        affinities = affinities.reshape(b, k, h, w)            # (B, K, H, W)
        # 区域 value / Region value
        region_v = self.W_value(context)                       # (B, K, out_ch)
        # 聚合: ( B, out _ ch, HW ) = ( B, out _ ch, K ) × ( B, K, HW ) / Aggregate: (B, out_ch, HW) = (B, out_ch, K) × (B, K, HW)
        ocr = torch.bmm(region_v.permute(0, 2, 1),
                        affinities.reshape(b, k, h * w))
        ocr = ocr.reshape(b, -1, h, w)                         # (B, out_ch, H, W)
        ocr = self.W_project(ocr)
        return self.dropout(ocr)


@DECODER_REGISTRY.register("ocrnet")
class OCRNetDecoder(nn.Module):
    """OCRNet decoder: SpatialGather + SpatialOCR + fusion.
        OCRNet 解码器。

    中文: OCRNet 解码器：空间聚合 + 空间 OCR + 融合。

    Operates on the deepest (bottleneck) feature only; external skip
    connections are IGNORED (``has_internal_skip = True``).

    Args:
        encoder_channels: list of encoder output channels (unused, kept for API).
        bottleneck_channels: bottleneck output channel count.
        num_classes: number of segmentation classes (default 19).
        ocr_mid_channels: intermediate channel count in OCR module (default 256).
        ocr_out_channels: OCR output channel count (default 512).
        dropout: dropout rate in OCR module (default 0.05).
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, num_classes: int = 19,
                 ocr_mid_channels: int = 256, ocr_out_channels: int = 512,
                 dropout: float = 0.05, **kwargs):
        super().__init__()
        in_ch = bottleneck_channels

        # Coarse 分割 头部 ( for SpatialGather soft 掩码 ) / Coarse segmentation head (for SpatialGather soft mask)
        self.coarse_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )

        # OCR 模块 / OCR module
        self.spatial_gather = _SpatialGather(scale=1.0)
        self.spatial_ocr = _SpatialOCR(
            in_ch=in_ch, key_ch=ocr_mid_channels,
            out_ch=ocr_out_channels, num_classes=num_classes,
            dropout=dropout,
        )

        # 融合: concat [ pixel _ feats, ocr _ feats ] → 1 × 1 conv → BN → ReLU / Fusion: concat [pixel_feats, ocr_feats] → 1×1 conv → BN → ReLU
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch + ocr_out_channels, ocr_out_channels, 1, bias=False),
            nn.BatchNorm2d(ocr_out_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = ocr_out_channels
        self._num_classes = num_classes

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]):
        """Forward pass.
            前向传播 pass。

        Args:
            bottleneck_feat: deepest feature after bottleneck (B, C, H, W).
            skip_features: list of skip features (IGNORED by OCR decoder).
        Returns:
            OCR-refined feature map (B, out_ch, H, W).
        """
        # Coarse 预测 → soft 掩码 / Coarse prediction → soft mask
        coarse = self.coarse_head(bottleneck_feat)           # (B, K, H, W)
        probs = F.softmax(coarse, dim=1)                     # (B, K, H, W)

        # SpatialGather: object-region representations
        context = self.spatial_gather(bottleneck_feat, probs) # (B, K, C)

        # SpatialOCR: OCR-augmented 特征 / SpatialOCR: OCR-augmented features
        ocr = self.spatial_ocr(bottleneck_feat, context)      # (B, out_ch, H, W)

        # 融合 original + OCR / Fuse original + OCR
        out = self.fusion(torch.cat([bottleneck_feat, ocr], dim=1))
        return out


__all__ = ['OCRNetDecoder']
