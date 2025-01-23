"""
This type stub file was generated by pyright.
"""

import numpy as np
import torch
from typing import BinaryIO, List, Optional, Tuple, Union
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform

SF_AUDIO_FILE_EXTENSIONS = ...
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = ...
def convert_waveform(waveform: Union[np.ndarray, torch.Tensor], sample_rate: int, normalize_volume: bool = ..., to_mono: bool = ..., to_sample_rate: Optional[int] = ...) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    ...

def get_waveform(path_or_fp: Union[str, BinaryIO], normalization: bool = ..., mono: bool = ..., frames: int = ..., start: int = ..., always_2d: bool = ..., output_sample_rate: Optional[int] = ..., normalize_volume: bool = ..., waveform_transforms: Optional[CompositeAudioWaveformTransform] = ...) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    ...

def get_features_from_npy_or_audio(path, waveform_transforms=...): # -> Any | ndarray[Any, Any]:
    ...

def get_features_or_waveform_from_stored_zip(path, byte_offset, byte_size, need_waveform=..., use_sample_rate=..., waveform_transforms=...): # -> Any | ndarray[Any, Any]:
    ...

def get_features_or_waveform(path: str, need_waveform=..., use_sample_rate=..., waveform_transforms=...): # -> ndarray[Any, Any] | Any:
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    ...

def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=..., waveform_transforms=...) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    ...

def is_npy_data(data: bytes) -> bool:
    ...

def is_sf_audio_data(data: bytes) -> bool:
    ...

def mmap_read(path: str, offset: int, length: int) -> bytes:
    ...

def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    ...

def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """
    ...

def get_window(window_fn: callable, n_fft: int, win_length: int) -> torch.Tensor:
    ...

def get_fourier_basis(n_fft: int) -> torch.Tensor:
    ...

def get_mel_filters(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    ...

class TTSSpectrogram(torch.nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int, window_fn: callable = ..., return_phase: bool = ...) -> None:
        ...
    
    def forward(self, waveform: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ...
    


class TTSMelScale(torch.nn.Module):
    def __init__(self, n_mels: int, sample_rate: int, f_min: float, f_max: float, n_stft: int) -> None:
        ...
    
    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        ...
    


