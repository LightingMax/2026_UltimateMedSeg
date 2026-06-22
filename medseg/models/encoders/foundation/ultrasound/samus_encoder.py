"""SAMUS ultrasound foundation-model encoder.
    SAMUS ultrasound foundation-model 编码器。

Reference:
    Xian Lin, Yangyang Xiang, Li Yu, Zengqiang Yan.
    "Beyond Adapting SAM: Towards End-to-End Ultrasound Image Segmentation
    via Auto Prompting." MICCAI 2024.
    arXiv: 2309.06824. DOI: 10.1007/978-3-031-72111-3_3
    Upstream code: https://github.com/xianlin7/SAMUS

SAMUS adapts SAM's ViT-B/16 image encoder for ultrasound by injecting a
learnable *position-bias adapter* between the positional embedding and the
transformer blocks.  The adapter compensates for the resolution mismatch
between SAM's native 1024x1024 pre-training and the 256x256 ultrasound
frames used in SAMUS.

This encoder extracts the SAMUS ViT-B backbone (with adapter) and projects
its intermediate block tokens into a 4-stage DPT-style multi-scale pyramid
(deepest LAST), matching the ``BaseFoundationEncoder`` contract.

Registered as ``"samus"`` in ``ENCODER_REGISTRY``.
"""
# Source: https://github.com/xianlin7/SAMUS

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY
from medseg.models.encoders.foundation._base import (
    DPTHead,
    BaseFoundationEncoder,
    load_with_ssl_fallback,
)


# ---------------------------------------------------------------------------
# Helpers (ported from networks/sam/samus.py)
# ---------------------------------------------------------------------------

def _interpolate_pos_embed(pos_embed: torch.Tensor, num_prefix: int,
                           new_hw: tuple) -> torch.Tensor:
    """Bicubic-resample a 1-D positional embedding to a new (H, W) grid.
        Bicubic-resample a 1-D positional 嵌入 to a new ( H, W ) grid。

    ``pos_embed`` has shape (1, num_prefix + N, C) where the first
    ``num_prefix`` tokens (e.g. the CLS token) are kept untouched and the
    remaining N entries form a square grid that is interpolated to the
    requested (H, W) shape.
    """
    prefix = pos_embed[:, :num_prefix]
    grid = pos_embed[:, num_prefix:]
    N = grid.shape[1]
    C = grid.shape[-1]
    old = int(round(math.sqrt(N)))
    if old * old != N:
        raise ValueError(
            'pos_embed grid is not square: N=%d (sqrt=%.3f)' % (N, math.sqrt(N))
        )
    new_h, new_w = new_hw
    grid = grid.reshape(1, old, old, C).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid, size=(new_h, new_w), mode='bicubic', align_corners=False,
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, C)
    return torch.cat([prefix, grid], dim=1)


class _PositionBiasAdapter(nn.Module):
    """Small MLP that injects a learnable bias into the patch tokens.
        Small MLP that injects a learnable 偏置 into the 图块 标记。

    SAMUS notes that SAM's positional embeddings are tuned for 1024x1024
    inputs and degrade on 256x256 ultrasound frames.  The adapter is a
    lightweight residual that lets the encoder shift its position-aware
    features without retraining the backbone.
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        hidden = max(int(dim * mlp_ratio), 8)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        # Zero-init the 输出 projection so the 适配器 starts as a no-op / Zero-init the output projection so the adapter starts as a no-op.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# ViT-B/16 wrapper with position-bias adapter
# ---------------------------------------------------------------------------

class _SAMUSViTWrapper(nn.Module):
    """Wrap a timm ViT-B/16 to expose the SAMUS forward (with adapter).
        封装 a timm ViT-B / 16 to 暴露 the SAMUS forward ( with 适配器 )。

    Provides ``get_intermediate_layers``, ``blocks``, ``embed_dim``,
    ``num_prefix_tokens`` and ``patch_embed.patch_size`` so that the
    ``BaseFoundationEncoder`` / ``DPTHead`` pipeline works transparently.
    """

    EMBED_DIM = 768
    PATCH_SIZE = 16

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        # Backbone is always 3-channel; input_adapter handles non-RGB inputs.
        vit = load_with_ssl_fallback(
            timm.create_model,
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )

        # 图块 嵌入: keep only the Conv2d projection so we are free to / Patch embedding: keep only the Conv2d projection so we are free to
        # feed inputs of arbitrary ( H, W ) that are multiples of 图块 _ 大小 / feed inputs of arbitrary (H, W) that are multiples of patch_size.
        self.proj = vit.patch_embed.proj
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed  # (1, 1 + 14*14, 768)
        self.pos_drop = getattr(vit, 'pos_drop', nn.Identity())
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Expose patch_embed.patch_size for introspection (BaseFoundationEncoder).
        class _PE:
            patch_size = (self.PATCH_SIZE, self.PATCH_SIZE)
        self.patch_embed = _PE()

        # Trainable position-bias 适配器 / Trainable position-bias adapter.
        self.pos_bias_adapter = _PositionBiasAdapter(self.EMBED_DIM)

        # Introspection fields required by BaseFoundationEncoder.
        self.embed_dim = self.EMBED_DIM
        self.num_prefix_tokens = 1  # CLS token
        self.num_features = self.EMBED_DIM

    def get_intermediate_layers(self, x: torch.Tensor, n) -> List[torch.Tensor]:
        """Extract token grids from specified blocks (timm-compatible).
            从指定 block 提取 token 网格 ( timm 兼容 )。

        Args:
            x: input image (B, C, H, W), already padded to patch multiple.
            n: list of block indices.

        Returns:
            List of (B, N_patches, C) tensors with prefix tokens stripped.
        """
        B = x.shape[0]
        # patch embed via Conv2d projection (avoids timm img_size assertion)
        x = self.proj(x)  # (B, C, Hp, Wp)
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + Hp*Wp, C)

        pos = _interpolate_pos_embed(self.pos_embed, num_prefix=1,
                                     new_hw=(Hp, Wp))
        x = x + pos
        x = self.pos_bias_adapter(x)
        x = self.pos_drop(x)

        target_set = set(n) if isinstance(n, (list, tuple)) else \
            set(range(len(self.blocks) - n, len(self.blocks)))

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in target_set:
                # 去掉 prefix tokens / Strip prefix tokens.
                outputs.append(x[:, self.num_prefix_tokens:, :])
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.get_intermediate_layers(x, n=[len(self.blocks) - 1])
        return outs[-1]


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("samus")
class SAMUSEncoder(BaseFoundationEncoder):
    """SAMUS ultrasound encoder (ViT-B/16 + position-bias adapter, ``embed_dim=768``).
        SAMUS ultrasound 编码器。

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    img_size : int
        Reference spatial size (default 256 to match the SAMUS paper).
    pretrained : bool
        Whether to load ImageNet-pretrained ViT-B/16 weights via timm.
    pretrained_path : Optional[str]
        Path to a local SAMUS checkpoint. Only encoder weights are loaded
        (keys starting with ``image_encoder.`` are stripped of that prefix).
    freeze / unfreeze_last_n / inference_only :
        Standard freeze controls inherited via :class:`FreezeMixin`.
    """

    native_img_size: int = 256
    PATCH_SIZE = _SAMUSViTWrapper.PATCH_SIZE
    EMBED_DIM = _SAMUSViTWrapper.EMBED_DIM

    def __init__(self, in_channels: int = 3, img_size: Optional[int] = None,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False, **kwargs):
        resolved_img_size = int(img_size) if img_size is not None else self.native_img_size
        super().__init__(in_channels=in_channels, img_size=resolved_img_size,
                         pretrained=pretrained, pretrained_path=pretrained_path,
                         freeze=freeze, unfreeze_last_n=unfreeze_last_n,
                         inference_only=inference_only, **kwargs)

        # 通道 适配器 for non-RGB inputs / Channel adapter for non-RGB inputs.
        if in_channels != 3:
            self.input_adapter: nn.Module = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        self.backbone = _SAMUSViTWrapper(
            in_channels=in_channels,
            pretrained=pretrained and pretrained_path is None,
        )

        # Load SAMUS-specific checkpoint if provided.
        if pretrained_path is not None:
            self._load_samus_checkpoint(pretrained_path)

        # Introspection.
        ps = self.backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            self.patch_size = int(ps[0])
        else:
            self.patch_size = int(ps)
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix_tokens = int(self.backbone.num_prefix_tokens)

        # DPT head: 从不同深度 block 构建真正多尺度金字塔
        # DPT 头部: genuine 多尺度 金字塔 from different-depth blocks / DPT head: genuine multi-scale pyramid from different-depth blocks
        self.dpt = DPTHead(
            embed_dim=self.embed_dim,
            num_prefix_tokens=int(self.num_prefix_tokens),
        )
        self.out_channels = self.dpt.out_channels
        self._block_indices = DPTHead.default_block_indices(len(self.backbone.blocks))

        self._maybe_inject_adapters()
        self._apply_freeze_policy()

    def _load_samus_checkpoint(self, path: str):
        """Load encoder weights from a SAMUS checkpoint.
            从 SAMUS 检查点 加载 编码器 权重。

        SAMUS checkpoints store the full model (encoder + decoder). We keep
        only keys that belong to the ViT backbone or the position-bias adapter
        and strip the ``image_encoder.`` prefix.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break

        if not isinstance(ckpt, dict):
            warnings.warn(f"[samus] checkpoint at '{path}' is not a dict, skipped.")
            return

        # Keep only encoder keys, strip the ``image_encoder.`` prefix.
        enc_prefix = "image_encoder."
        cleaned = {}
        for k, v in ckpt.items():
            nk = k
            if nk.startswith(enc_prefix):
                nk = nk[len(enc_prefix):]
            else:
                # Skip decoder / prompt-encoder keys.
                if any(skip in k for skip in ("mask_decoder", "prompt_encoder",
                                              "decoder", "neck")):
                    continue
            cleaned[nk] = v

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        n_loaded = len(cleaned) - len(unexpected)
        print(f"[samus] loaded {n_loaded} encoder weights from '{path}' "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_adapter(x)
        B, _, H, W = x.shape
        p = self.patch_size

        # 填充到 patch_size 的倍数 / Pad to multiple of patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # 从不同深度 block 提取 token（DPT 核心）
        # 提取 标记 from different-depth blocks ( DPT core ) / Extract tokens from different-depth blocks (DPT core)
        multi_tokens = self.backbone.get_intermediate_layers(
            x, n=self._block_indices,
        )

        h_patches = Hp // p
        w_patches = Wp // p

        return self.dpt(list(multi_tokens), h_patches, w_patches, H, W)
