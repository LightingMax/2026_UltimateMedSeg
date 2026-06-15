"""KAN / MLP / LSTM-based complete 分割 architectures。
    KAN / MLP / LSTM-based complete segmentation architectures."""

from .ukan import UKAN
from .wav_kan_unet import WavKANUNet

__all__ = ["UKAN", "WavKANUNet"]
