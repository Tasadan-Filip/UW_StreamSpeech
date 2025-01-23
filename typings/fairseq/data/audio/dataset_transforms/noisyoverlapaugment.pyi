"""
This type stub file was generated by pyright.
"""

from fairseq.data.audio.dataset_transforms import AudioDatasetTransform, register_audio_dataset_transform

_DEFAULTS = ...
@register_audio_dataset_transform("noisyoverlapaugment")
class NoisyOverlapAugment(AudioDatasetTransform):
    @classmethod
    def from_config_dict(cls, config=...): # -> NoisyOverlapAugment:
        ...
    
    def __init__(self, rate=..., mixing_noise_rate=..., noise_path=..., noise_snr_min=..., noise_snr_max=..., utterance_snr_min=..., utterance_snr_max=...) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __call__(self, sources):
        ...
    


