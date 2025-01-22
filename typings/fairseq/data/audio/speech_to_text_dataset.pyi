"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from fairseq.data import Dictionary, FairseqDataset
from fairseq.data.audio.data_cfg import S2TDataConfig

logger = ...
@dataclass
class SpeechToTextDatasetItem:
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = ...
    speaker_id: Optional[int] = ...


class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = ...
    def __init__(self, split: str, is_train_split: bool, cfg: S2TDataConfig, audio_paths: List[str], n_frames: List[int], src_texts: Optional[List[str]] = ..., tgt_texts: Optional[List[str]] = ..., speakers: Optional[List[str]] = ..., src_langs: Optional[List[str]] = ..., tgt_langs: Optional[List[str]] = ..., ids: Optional[List[str]] = ..., tgt_dict: Optional[Dictionary] = ..., pre_tokenizer=..., bpe_tokenizer=..., n_frames_per_step=..., speaker_to_id=..., append_eos=...) -> None:
        ...
    
    def get_tgt_lens_and_check_oov(self): # -> list[int] | list[Any]:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    @classmethod
    def is_lang_tag(cls, token): # -> Match[str] | None:
        ...
    
    def check_tgt_lang_tag(self): # -> None:
        ...
    
    @classmethod
    def tokenize(cls, tokenizer, text: str): # -> str:
        ...
    
    def get_tokenized_tgt_text(self, index: int): # -> str:
        ...
    
    def pack_frames(self, feature: torch.Tensor): # -> Tensor:
        ...
    
    @classmethod
    def get_lang_tag_idx(cls, lang: str, dictionary: Dictionary): # -> int:
        ...
    
    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def collater(self, samples: List[SpeechToTextDatasetItem], return_order: bool = ...) -> Dict:
        ...
    
    def num_tokens(self, index):
        ...
    
    def size(self, index): # -> tuple[Any, Any]:
        ...
    
    @property
    def sizes(self): # -> NDArray[Any]:
        ...
    
    @property
    def can_reuse_epoch_itr_across_epochs(self): # -> Literal[True]:
        ...
    
    def ordered_indices(self): # -> Any:
        ...
    
    def prefetch(self, indices):
        ...
    


class SpeechToTextDatasetCreator:
    KEY_TGT_TEXT = ...
    DEFAULT_LANG = ...
    @classmethod
    def get_size_ratios(cls, datasets: List[SpeechToTextDataset], alpha: float = ...) -> List[float]:
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        ...
    
    @classmethod
    def from_tsv(cls, root: str, cfg: S2TDataConfig, splits: str, tgt_dict, pre_tokenizer, bpe_tokenizer, is_train_split: bool, epoch: int, seed: int, n_frames_per_step: int = ..., speaker_to_id=...) -> SpeechToTextDataset:
        ...
    


