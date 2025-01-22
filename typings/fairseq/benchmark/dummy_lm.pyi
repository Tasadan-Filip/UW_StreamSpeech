"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Optional
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

logger = ...
@dataclass
class DummyLMConfig(FairseqDataclass):
    dict_size: int = ...
    dataset_size: int = ...
    tokens_per_sample: int = ...
    add_bos_token: bool = ...
    batch_size: Optional[int] = ...
    max_tokens: Optional[int] = ...
    max_target_positions: int = ...


@register_task("dummy_lm", dataclass=DummyLMConfig)
class DummyLMTask(FairseqTask):
    def __init__(self, cfg: DummyLMConfig) -> None:
        ...
    
    def load_dataset(self, split, epoch=..., combine=..., **kwargs): # -> None:
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        ...
    
    @property
    def source_dictionary(self): # -> Dictionary:
        ...
    
    @property
    def target_dictionary(self): # -> Dictionary:
        ...
    


