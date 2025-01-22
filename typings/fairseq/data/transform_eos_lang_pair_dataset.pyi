"""
This type stub file was generated by pyright.
"""

from typing import Optional
from . import FairseqDataset

class TransformEosLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """
    def __init__(self, dataset: FairseqDataset, src_eos: int, new_src_eos: Optional[int] = ..., tgt_bos: Optional[int] = ..., new_tgt_bos: Optional[int] = ...) -> None:
        ...
    
    def __getitem__(self, index):
        ...
    
    def __len__(self): # -> int:
        ...
    
    def collater(self, samples, **extra_args):
        ...
    
    def num_tokens(self, index):
        ...
    
    def size(self, index):
        ...
    
    @property
    def sizes(self):
        ...
    
    def ordered_indices(self): # -> NDArray[signedinteger[_64Bit]]:
        ...
    
    @property
    def supports_prefetch(self): # -> Any | bool:
        ...
    
    def prefetch(self, indices):
        ...
    


