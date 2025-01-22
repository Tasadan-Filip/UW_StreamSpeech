from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import importlib
import os
import numpy as np

from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetItem,
)
import torch


class AudioTransform(ABC):
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        pass


class CompositeAudioTransform(AudioTransform):
    def _from_config_dict(
        cls,
        transform_type,
        get_audio_transform,
        composite_cls,
        config=None,
        return_empty=False,
    ):
        _config = {} if config is None else config
        _transforms = _config.get(f"{transform_type}_transforms")

        if _transforms is None:
            if return_empty:
                _transforms = []
            else:
                return None

        transforms = [
            get_audio_transform(_t).from_config_dict(_config.get(_t))
            for _t in _transforms
        ]
        return composite_cls(transforms)

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = (
            [self.__class__.__name__ + "("]
            + [f"    {t.__repr__()}" for t in self.transforms]
            + [")"]
        )
        return "\n".join(format_string)


def register_audio_transform(name, cls_type, registry, class_names):
    def register_audio_transform_cls(cls):
        if name in registry:
            raise ValueError(f"Cannot register duplicate transform ({name})")
        if not issubclass(cls, cls_type):
            raise ValueError(
                f"Transform ({name}: {cls.__name__}) must extend "
                f"{cls_type.__name__}"
            )
        if cls.__name__ in class_names:
            raise ValueError(
                f"Cannot register audio transform with duplicate "
                f"class name ({cls.__name__})"
            )
        registry[name] = cls
        class_names.add(cls.__name__)
        return cls

    return register_audio_transform_cls


def import_transforms(transforms_dir, transform_type):
    for file in os.listdir(transforms_dir):
        path = os.path.join(transforms_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(
                f"fairseq.data.audio.{transform_type}_transforms." + name
            )


# Utility fn for uniform numbers in transforms
def rand_uniform(a, b):
    return np.random.uniform() * (b - a) + a


class AudioWaveformTransform(AudioTransform):
    pass


AUDIO_WAVEFORM_TRANSFORM_REGISTRY = {}
AUDIO_WAVEFORM_TRANSFORM_CLASS_NAMES = set()


def get_audio_waveform_transform(name):
    return AUDIO_WAVEFORM_TRANSFORM_REGISTRY[name]


def register_audio_waveform_transform(name):
    return register_audio_transform(
        name,
        AudioWaveformTransform,
        AUDIO_WAVEFORM_TRANSFORM_REGISTRY,
        AUDIO_WAVEFORM_TRANSFORM_CLASS_NAMES,
    )


import_transforms(os.path.dirname(__file__), "waveform")


class CompositeAudioWaveformTransform(CompositeAudioTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        return super()._from_config_dict(
            cls,
            "waveform",
            get_audio_waveform_transform,
            CompositeAudioWaveformTransform,
            config,
        )

    def __call__(self, x, sample_rate):
        for t in self.transforms:
            x, sample_rate = t(x, sample_rate)
        return x, sample_rate
    
class SpeechToTextMultitaskDataset(SpeechToTextDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]:
        s2t_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2t_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToTextDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out