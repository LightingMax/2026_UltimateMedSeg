"""TransUNet – self-contained port from github.com/Beckschen/TransUNet.
    TransUNet – self-contained 移植 from github. com / Beckschen / TransUNet。

Combines vit_seg_configs.py, vit_seg_modeling_resnet_skip.py, and
vit_seg_modeling.py into a single file with no external medseg imports.

Standard interface:
    model = TransUNet(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/Beckschen/TransUNet

from __future__ import absolute_import, division, print_function

import copy
import logging
import math
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair

logger = logging.getLogger(__name__)


# ─ ─ 轻量级 配置 ( replaces ml _ collections. ConfigDict ) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ / ── lightweight config (replaces ml_collections.ConfigDict) ──────────────────
class _Cfg(dict):
    """Dict that also supports attribute access."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


def _get_r50_b16_config():
    c = _Cfg()
    c.patches = _Cfg({"grid": (16, 16)})
    c.hidden_size = 768
    c.transformer = _Cfg(mlp_dim=3072, num_heads=12, num_layers=12,
                         attention_dropout_rate=0.0, dropout_rate=0.1)
    c.resnet = _Cfg(num_layers=(3, 4, 9), width_factor=1)
    c.classifier = "seg"
    c.decoder_channels = (256, 128, 64, 16)
    c.skip_channels = [512, 256, 64, 16]
    c.n_classes = 2
    c.n_skip = 3
    c.activation = "softmax"
    return c


# ── ResNetV2 with 跳跃连接 / ── ResNetV2 with skip connections (from vit_seg_modeling_resnet_skip.py) ───
def _np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


# JAX checkpoint key-name constants (from official TransUNet source)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                     bias=bias, groups=groups)


def _conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0,
                     bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        """Load weights from JAX checkpoint (faithful to official TransUNet)."""
        conv1_weight = _np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = _np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = _np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = _np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = _np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = _np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = _np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = _np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = _np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))
        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))
        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = _np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = _np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = _np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False,
                               padding=3)),
            ("gn", nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width, cout=width * 4,
                                            cmid=width))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 4, cout=width * 4,
                                               cmid=width))
                 for i in range(2, block_units[0] + 1)]))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 4, cout=width * 8,
                                            cmid=width * 2, stride=2))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 8, cout=width * 8,
                                               cmid=width * 2))
                 for i in range(2, block_units[1] + 1)]))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 8, cout=width * 16,
                                            cmid=width * 4, stride=2))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 16, cout=width * 16,
                                               cmid=width * 4))
                 for i in range(2, block_units[2] + 1)]))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                feat = torch.zeros((b, x.size()[1], right_size, right_size),
                                   device=x.device)
                feat[:, :, : x.size()[2], : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ── ViT building blocks (from vit_seg_modeling.py) ──────────────────────────
def _swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": _swish}


class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads,
                                      self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size)
        probs = self.softmax(scores)
        weights = probs if self.vis else None
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.size()[:-2], self.all_head_size)
        out = self.proj_dropout(self.out(ctx))
        return out, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = ((img_size[0] // patch_size_real[0]) *
                         (img_size[1] // patch_size_real[1]))
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = ((img_size[0] // patch_size[0]) *
                         (img_size[1] // patch_size[1]))
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return self.dropout(embeddings), features


class Block(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x, weights = self.attn(self.attention_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x))
        return x + h, weights

    def load_from(self, weights, n_block):
        """Load weights from JAX checkpoint (faithful to official TransUNet)."""
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = _np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            key_weight = _np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            value_weight = _np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            out_weight = _np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()

            query_bias = _np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = _np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = _np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = _np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = _np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = _np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = _np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = _np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                _np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(
                _np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(
                _np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(
                _np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([
            copy.deepcopy(Block(config, vis))
            for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        emb, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(emb)
        return encoded, attn_weights, features


# ── Decoder (CUP) ────────────────────────────────────────────────────────────
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1,
                 use_batchnorm=True):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_ch)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, 3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_ch, out_ch, 3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        up = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
              if upsampling > 1 else nn.Identity())
        super().__init__(conv2d, up)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3,
                                    padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if config.n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        blocks = [DecoderBlock(ic, oc, sc)
                  for ic, oc, sc in zip(in_channels, out_channels,
                                        skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = (features[i] if (features is not None and
                                    i < self.config.n_skip) else None)
            x = decoder_block(x, skip=skip)
        return x


# ── Top-level model ──────────────────────────────────────────────────────────
class _VisionTransformer(nn.Module):
    """Original TransUNet VisionTransformer."""
    def __init__(self, config, img_size=224, num_classes=2, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"], kernel_size=3)
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)

    def load_from(self, weights):
        """Load weights from JAX checkpoint (faithful to official TransUNet)."""
        from scipy import ndimage
        with torch.no_grad():
            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                _np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                _np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(
                _np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                _np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = _np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                logger.info('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid.numpy(), zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                self.transformer.embeddings.position_embeddings.copy_(
                    torch.from_numpy(posemb_grid))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    _np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = _np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = _np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class TransUNet(nn.Module):
    """TransUNet wrapper with standard interface.
        TransUNet 封装器 with 标准 interface。

    Args:
        in_channels (int): Number of input channels (default: 3).
        num_classes (int): Number of output classes (default: 2).
        img_size (int): Input image size (default: 224).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        pretrained = kwargs.pop("pretrained", True)
        pretrained_path = kwargs.pop("pretrained_path", None)
        config = _get_r50_b16_config()
        config.n_classes = num_classes
        # 计算 grid dynamically based on img _ 大小 / Compute grid dynamically based on img_size
        grid = img_size // 16  # 14 for 224, 16 for 256
        config.patches["grid"] = (grid, grid)
        # Allow kwargs to 覆盖 配置 / Allow kwargs to override config
        if "hidden_size" in kwargs:
            config.hidden_size = kwargs["hidden_size"]
        if "num_layers" in kwargs:
            config.transformer["num_layers"] = kwargs["num_layers"]
        if "num_heads" in kwargs:
            config.transformer["num_heads"] = kwargs["num_heads"]
        if "mlp_dim" in kwargs:
            config.transformer["mlp_dim"] = kwargs["mlp_dim"]
        if "resnet_num_layers" in kwargs:
            config.resnet["num_layers"] = tuple(kwargs["resnet_num_layers"])
        if "grid_size" in kwargs:
            config.patches["grid"] = tuple(kwargs["grid_size"])
        if "decoder_channels" in kwargs:
            config.decoder_channels = tuple(kwargs["decoder_channels"])
        if "skip_channels" in kwargs:
            config.skip_channels = list(kwargs["skip_channels"])
        if "n_skip" in kwargs:
            config.n_skip = kwargs["n_skip"]
        self.model = _VisionTransformer(config, img_size=img_size,
                                        num_classes=num_classes)

        if pretrained:
            self._load_jax_pretrained(pretrained_path)

    def _load_jax_pretrained(self, pretrained_path=None):
        """Load R50+ViT-B/16 JAX checkpoint (faithful to official TransUNet).

        Delegates key remapping to _VisionTransformer.load_from, which mirrors
        the official repository's load_from method exactly.
        """
        from medseg.utils.weight_downloader import ensure_weight

        weight_path = pretrained_path
        if weight_path is None:
            weight_path = str(ensure_weight("transunet_r50_vit_b16"))

        try:
            npz = np.load(weight_path)
            self.model.load_from(npz)
            logger.info("TransUNet: loaded JAX pretrained from %s", weight_path)
        except Exception as e:
            import warnings
            warnings.warn(
                f"TransUNet: failed to load pretrained weights from "
                f"{weight_path}: {e}. Model initialized from scratch.")

    def forward(self, x):
        return self.model(x)
