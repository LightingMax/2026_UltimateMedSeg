"""GCN library for G-CASCADE decoder (graph convolution operations).
    GCN library for G-CASCADE 解码器。

Adapted from ViG (Vision GNN) graph convolution primitives.
Source: https://github.com/SLDGroup/G-CASCADE
"""

from .torch_vertex import Grapher

__all__ = ["Grapher"]
