"""
This type stub file was generated by pyright.
"""

from typing import Dict, Optional, Tuple
from torch import Tensor, nn
from fairseq.incremental_decoding_utils import with_incremental_state

has_megatron_submodule = ...
@with_incremental_state
class ModelParallelMultiheadAttention(nn.Module):
    """Model parallel Multi-headed attention.
    This performs the Multi-headed attention over multiple gpus.

    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """
    def __init__(self, embed_dim, num_heads, kdim=..., vdim=..., dropout=..., bias=..., self_attention=..., encoder_decoder_attention=...) -> None:
        ...
    
    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor], key_padding_mask: Optional[Tensor] = ..., incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = ..., static_kv: bool = ..., attn_mask: Optional[Tensor] = ..., **unused_kwargs) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        ...
    
    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order): # -> Dict[str, Dict[str, Tensor | None]]:
        """Reorder buffered internal state (for incremental generation)."""
        ...
    


