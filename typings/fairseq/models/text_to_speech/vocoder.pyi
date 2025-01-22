"""
This type stub file was generated by pyright.
"""

import torch
from typing import Dict
from torch import nn
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig

logger = ...
class PseudoInverseMelScale(torch.nn.Module):
    def __init__(self, n_stft, n_mels, sample_rate, f_min, f_max) -> None:
        ...
    
    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        ...
    


class GriffinLim(torch.nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int, n_iter: int, window_fn=...) -> None:
        ...
    
    @classmethod
    def get_window_sum_square(cls, n_frames, hop_length, win_length, n_fft, window_fn=...) -> torch.Tensor:
        ...
    
    def inverse(self, magnitude: torch.Tensor, phase) -> torch.Tensor:
        ...
    
    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        ...
    


class GriffinLimVocoder(nn.Module):
    def __init__(self, sample_rate, win_size, hop_size, n_fft, n_mels, f_min, f_max, window_fn, spec_bwd_max_iter=..., fp16=...) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    
    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig): # -> Self:
        ...
    


class HiFiGANVocoder(nn.Module):
    def __init__(self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = ...) -> None:
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    @classmethod
    def from_data_cfg(cls, args, data_cfg: S2TDataConfig): # -> Self:
        ...
    


class CodeHiFiGANVocoder(nn.Module):
    def __init__(self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = ...) -> None:
        ...
    
    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=...) -> torch.Tensor:
        ...
    
    @classmethod
    def from_data_cfg(cls, args, data_cfg): # -> Self:
        ...
    


def get_vocoder(args, data_cfg: S2TDataConfig): # -> GriffinLimVocoder | HiFiGANVocoder | CodeHiFiGANVocoder:
    ...

