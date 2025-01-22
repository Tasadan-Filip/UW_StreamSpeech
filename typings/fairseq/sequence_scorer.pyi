"""
This type stub file was generated by pyright.
"""

import torch

class SequenceScorer:
    """Scores the target for a given source sentence."""
    def __init__(self, tgt_dict, softmax_batch=..., compute_alignment=..., eos=..., symbols_to_strip_from_output=...) -> None:
        ...
    
    @torch.no_grad()
    def generate(self, models, sample, **kwargs): # -> list[Any]:
        """Score a batch of translations."""
        ...
    


