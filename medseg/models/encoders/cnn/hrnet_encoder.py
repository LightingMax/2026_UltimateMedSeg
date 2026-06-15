"""HRNet encoder: High-Resolution Network backbone for multi-scale feature extraction.
    HRNet 编码器。

中文: HRNet 编码器：高分辨率网络骨干，用于多尺度特征提取。

Wraps the HRNet backbone (Sun et al., CVPR 2019) as a multi-scale feature
extractor. HRNet is unique among encoder families: it maintains parallel
branches at multiple resolutions throughout the network, fusing features
across resolutions at every stage. This preserves high-resolution spatial
detail that other encoders lose through progressive downsampling.

Registered encoders:
    - ``hrnet_w18``: HRNet-W18 (lightweight, channels [18, 36, 72, 144])
    - ``hrnet_w32``: HRNet-W32 (standard, channels [32, 64, 128, 256])

Returns 4 feature maps at strides {4, 8, 16, 32} with the deepest LAST.

Reference:
    Sun et al., "Deep High-Resolution Representation Learning for Human
    Pose Estimation." CVPR 2019.
    https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
"""
# Source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

from typing import List

import torch
import torch.nn as nn

from medseg.registry import ENCODER_REGISTRY


def _get_hrnet_backbone(in_channels, width):
    """Lazy import to avoid 循环的 dependency ( encoders 加载 before networks )。
        Lazy import to avoid circular dependency (encoders load before networks)."""
    from medseg.models.networks.cnn.hrnet_model import _HRNetBackbone
    return _HRNetBackbone(in_channels, width)


def _register_hrnet_encoder(name: str, width: int):
    """Register an HRNet variant as an 编码器。
        Register an HRNet variant as an encoder."""
    w = width
    channels = [w, w * 2, w * 4, w * 8]

    @ENCODER_REGISTRY.register(name)
    class HRNetEncoder(nn.Module):
        __doc__ = (
            f"HRNet encoder producing 4 multi-scale features at strides "
            f"{{4, 8, 16, 32}}. out_channels = {channels}. "
            f"Forward returns a list with the deepest feature LAST."
        )

        def __init__(self, in_channels: int = 3, img_size: int = 224,
                     pretrained: bool = False, **kwargs):
            super().__init__()
            self.backbone = _get_hrnet_backbone(in_channels, width)
            self.out_channels: List[int] = list(channels)
            self.img_size = img_size

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            feats = self.backbone(x)
            # feats: [w@/4, 2w@/8, 4w@/16, 8w@/32] — deepest LAST
            return feats

    HRNetEncoder.__name__ = f"HRNetEncoder_W{width}"
    HRNetEncoder.__qualname__ = f"HRNetEncoder_W{width}"


_register_hrnet_encoder("hrnet_w18", 18)
_register_hrnet_encoder("hrnet_w32", 32)
