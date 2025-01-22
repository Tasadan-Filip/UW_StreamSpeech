"""
This type stub file was generated by pyright.
"""

import torch
from typing import Dict, Optional, Tuple
from torch import Tensor, nn
from fairseq.incremental_decoding_utils import with_incremental_state

_xformers_available = ...
@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, kdim=..., vdim=..., dropout=..., bias=..., add_bias_kv=..., add_zero_attn=..., self_attention=..., encoder_decoder_attention=..., q_noise=..., qn_block_size=..., xformers_att_config: Optional[str] = ..., xformers_blocksparse_layout: Optional[torch.Tensor] = ..., xformers_blocksparse_blocksize: Optional[int] = ...) -> None:
        ...
    
    def prepare_for_onnx_export_(self): # -> None:
        ...
    
    def reset_parameters(self): # -> None:
        ...
    
    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor], key_padding_mask: Optional[Tensor] = ..., incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = ..., need_weights: bool = ..., static_kv: bool = ..., attn_mask: Optional[Tensor] = ..., before_softmax: bool = ..., need_head_weights: bool = ...) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        ...
    
    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor): # -> Dict[str, Dict[str, Tensor | None]]:
        """Reorder buffered internal state (for incremental generation)."""
        ...
    
    def set_beam_size(self, beam_size): # -> None:
        """Used for effiecient beamable enc-dec attention"""
        ...
    
    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        ...
    
    def upgrade_state_dict_named(self, state_dict, name): # -> None:
        ...
    


