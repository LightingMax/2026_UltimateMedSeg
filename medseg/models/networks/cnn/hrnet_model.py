"""HRNet: Deep High-Resolution Representation Learning for Visual Recognition.
    HRNet: 深度 High-Resolution Representation Learning for Visual Recognition。

中文: HRNet：高分辨率表征学习网络。

Reference:
    Sun et al., "Deep High-Resolution Representation Learning for Human
    Pose Estimation." CVPR 2019.
    Upstream code: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

Architecture overview:
    HRNet maintains high-resolution representations throughout the network.
    It starts from a high-resolution sub-network, then adds parallel
    low-resolution sub-networks one by one, connected by multi-resolution
    fusion at every stage. The key innovation is that high-resolution
    features are never downsampled — instead, lower-resolution branches
    are maintained in parallel and fused back into the high-resolution
    branch via bilinear upsampling.

    This implementation provides HRNet-W18 and HRNet-W32 variants for
    2D medical image segmentation.

Self-contained: only torch is required.
"""
# Source: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    """标准 2-layer 残差 块 ( 3 × 3 conv × 2 )。
        Standard 2-layer residual block (3×3 conv × 2)."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _Bottleneck(nn.Module):
    """瓶颈层 块 ( 1 × 1 → 3 × 3 → 1 × 1 conv )。
        Bottleneck block (1×1 → 3×3 → 1×1 conv)."""
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(block, in_ch, out_ch, num_blocks, stride=1):
    """Build a 残差 层, with 可选 步长 - 1 下采样 on the first 块。
        Build a residual layer, with optional stride-1 downsampling on the first block."""
    downsample = None
    if stride != 1 or in_ch != out_ch * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * block.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch * block.expansion),
        )
    layers = [block(in_ch, out_ch, stride=stride, downsample=downsample)]
    in_ch = out_ch * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(in_ch, out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Multi-resolution 融合 分支 / Multi-resolution fusion branch
# ---------------------------------------------------------------------------
class _FuseBranch(nn.Module):
    """Fuse features from source branch j into target branch i.
        融合 特征 from 来源 分支 j into 目标 分支 i。

    - j < i (higher-res source → lower-res target): stride-2 3×3 conv, possibly
      stacked to achieve the correct downsampling factor.
    - j == i: identity.
    - j > i (lower-res source → higher-res target): 1×1 conv + bilinear upsample.
    """

    def __init__(self, in_ch, out_ch, i, j):
        super().__init__()
        self.i = i
        self.j = j
        if i == j:
            self.op = nn.Identity()
        elif i < j:
            # 上采样: 1 × 1 conv to match 通道 + bilinear / Upsample: 1×1 conv to match channels + bilinear
            self.op = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            # 下采样: chain 步长 - 2 3 × 3 convs / Downsample: chain stride-2 3×3 convs
            ops = []
            for k in range(i - j):
                c_in = in_ch if k == 0 else out_ch
                ops.extend([
                    nn.Conv2d(c_in, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
            # Last one without ReLU ( to allow 残差 addition ) / Last one without ReLU (to allow residual addition)
            ops[-1:] = [nn.BatchNorm2d(out_ch)]
            self.op = nn.Sequential(*ops)

    def forward(self, x, target_size):
        if self.i > self.j:
            # 下采样 / Downsample
            return self.op(x)
        elif self.i < self.j:
            # 上采样: 1 × 1 conv then bilinear / Upsample: 1×1 conv then bilinear
            return F.interpolate(self.op(x), size=target_size,
                                 mode='bilinear', align_corners=False)
        return x


class _HRModule(nn.Module):
    """One HRNet 模块: 并行的 branches + multi-resolution 融合。
        One HRNet module: parallel branches + multi-resolution fusion."""

    def __init__(self, num_branches, block, num_blocks, num_channels):
        super().__init__()
        self.num_branches = num_branches

        # 并行的 branches / Parallel branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            ch = num_channels[i]
            layers = []
            for _ in range(num_blocks[i]):
                layers.append(block(ch, ch))
            self.branches.append(nn.Sequential(*layers))

        # 融合 layers / Fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_ops = nn.ModuleList()
            for j in range(num_branches):
                fuse_ops.append(_FuseBranch(num_channels[j], num_channels[i], i, j))
            self.fuse_layers.append(fuse_ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        # Run 并行的 branches / Run parallel branches
        for i in range(self.num_branches):
            x_list[i] = self.branches[i](x_list[i])

        # Multi-resolution 融合 / Multi-resolution fusion
        fused = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                if i == j:
                    contrib = x_list[j]
                else:
                    contrib = self.fuse_layers[i][j](
                        x_list[j], x_list[i].shape[2:])
                y = contrib if y is None else y + contrib
            fused.append(self.relu(y))
        return fused


# ---------------------------------------------------------------------------
# HRNet 骨干网络 / HRNet backbone
# ---------------------------------------------------------------------------
class _HRNetBackbone(nn.Module):
    """HRNet backbone producing 4 parallel-resolution feature maps.
        HRNet 骨干网络 producing 4 parallel-resolution 特征图。

    Args:
        in_channels: input channels (default 3).
        width: base width for W18 (18) or W32 (32).
    """

    def __init__(self, in_channels=3, width=18):
        super().__init__()
        w = width
        # 通道 counts per 分支 at each 阶段 / Channel counts per branch at each stage
        self.stage2_channels = [w, w * 2]
        self.stage3_channels = [w, w * 2, w * 4]
        self.stage4_channels = [w, w * 2, w * 4, w * 8]

        # 主干: 2 × 步长 - 2 conv → / 4 分辨率 / Stem: 2× stride-2 conv → /4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # Stage 1: 瓶颈层 / Stage 1: Bottleneck blocks at /4 (produces 256 channels)
        self.stage1 = _make_layer(_Bottleneck, 64, 64, 4)  # out: 256

        # Transition 1: split into 2 branches (/4, /8)
        self.trans1 = nn.ModuleList([
            nn.Sequential(  # /4 branch: 256 → w
                nn.Conv2d(256, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w), nn.ReLU(inplace=True)),
            nn.Sequential(  # /8 branch: 256 → 2w
                nn.Conv2d(256, w * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True)),
        ])

        # 阶段 2: 2 branches, 1 模块 / Stage 2: 2 branches, 1 module
        self.stage2 = nn.ModuleList([
            _HRModule(2, _BasicBlock, [4, 4], self.stage2_channels),
        ])

        # Transition 2: add / 16 分支 / Transition 2: add /16 branch
        self.trans2 = nn.Sequential(
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),
        )

        # 阶段 3: 3 branches, 4 modules / Stage 3: 3 branches, 4 modules
        self.stage3 = nn.ModuleList([
            _HRModule(3, _BasicBlock, [4, 4, 4], self.stage3_channels)
            for _ in range(4)
        ])

        # Transition 3: add / 32 分支 / Transition 3: add /32 branch
        self.trans3 = nn.Sequential(
            nn.Conv2d(w * 4, w * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w * 8), nn.ReLU(inplace=True),
        )

        # 阶段 4: 4 branches, 3 modules / Stage 4: 4 branches, 3 modules
        self.stage4 = nn.ModuleList([
            _HRModule(4, _BasicBlock, [4, 4, 4, 4], self.stage4_channels)
            for _ in range(3)
        ])

    def forward(self, x):
        x = self.stem(x)       # /4
        x = self.stage1(x)     # /4, 256ch

        # Transition 1: split into 2 branches
        x_list = [self.trans1[0](x), self.trans1[1](x)]

        # 阶段 2 / Stage 2
        for module in self.stage2:
            x_list = module(x_list)

        # Transition 2: add 3rd 分支 / Transition 2: add 3rd branch
        x_list.append(self.trans2(x_list[-1]))

        # 阶段 3 / Stage 3
        for module in self.stage3:
            x_list = module(x_list)

        # Transition 3: add 4th 分支 / Transition 3: add 4th branch
        x_list.append(self.trans3(x_list[-1]))

        # 阶段 4 / Stage 4
        for module in self.stage4:
            x_list = module(x_list)

        return x_list  # [w@/4, 2w@/8, 4w@/16, 8w@/32]


# ---------------------------------------------------------------------------
# HRNet 分割 模型 / HRNet segmentation model
# ---------------------------------------------------------------------------
class HRNet(nn.Module):
    """HRNet segmentation model with multi-resolution fusion head.
        HRNet 分割 模型 with multi-resolution 融合 头部。

    中文: HRNet 分割网络，使用多分辨率融合头。

    Fuses all 4 parallel branches by upsampling to the highest resolution,
    concatenating, then classifying via a 1×1 conv head.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial size (default 224, unused).
        width: HRNet width — 18 for HRNet-W18, 32 for HRNet-W32 (default 18).
        deep_supervision: If True, adds auxiliary heads on lower branches
            during training (default False).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 width=18, deep_supervision=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        w = width
        self.backbone = _HRNetBackbone(in_channels, width=w)

        # Total fused 通道 when concatenating all 4 branches at / 4 / Total fused channels when concatenating all 4 branches at /4
        total_ch = w + w * 2 + w * 4 + w * 8  # W18: 480, W32: 800
        self.last_layer = nn.Sequential(
            nn.Conv2d(total_ch, total_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(total_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_ch, num_classes, 1),
        )

        # 深度 supervision heads on lower-resolution branches / Deep supervision heads on lower-resolution branches
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(w * 2, num_classes, 1),
                ),   # branch 1 (/8)
                nn.Sequential(
                    nn.Conv2d(w * 4, num_classes, 1),
                ),   # branch 2 (/16)
            ])

    def forward(self, x):
        _, _, H_in, W_in = x.shape
        feats = self.backbone(x)  # [w@/4, 2w@/8, 4w@/16, 8w@/32]

        # 上采样 all branches to the / 4 分辨率 / Upsample all branches to the /4 resolution
        target_size = feats[0].shape[2:]
        f0 = feats[0]
        f1 = F.interpolate(feats[1], size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(feats[2], size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(feats[3], size=target_size, mode='bilinear', align_corners=False)

        # 拼接 and classify / Concatenate and classify
        out = self.last_layer(torch.cat([f0, f1, f2, f3], dim=1))
        # 上采样 to original 输入 大小 / Upsample to original input size
        out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)

        if self.training and self.deep_supervision:
            aux = [
                F.interpolate(self.ds_heads[0](feats[1]),
                              size=(H_in, W_in), mode='bilinear', align_corners=False),
                F.interpolate(self.ds_heads[1](feats[2]),
                              size=(H_in, W_in), mode='bilinear', align_corners=False),
            ]
            return [out] + aux
        return out


__all__ = ['HRNet']
