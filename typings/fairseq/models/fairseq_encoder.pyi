"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

EncoderOut = ...
class FairseqEncoder(nn.Module):
    """Base class for encoders."""
    def __init__(self, dictionary) -> None:
        ...
    
    def forward(self, src_tokens, src_lengths=..., **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        ...
    
    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        ...
    
    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        ...
    
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        ...
    
    def max_positions(self): # -> float:
        """Maximum input length supported by the encoder."""
        ...
    
    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        ...
    
    def set_num_updates(self, num_updates): # -> None:
        """State from trainer to pass along to model at every update."""
        ...
    


