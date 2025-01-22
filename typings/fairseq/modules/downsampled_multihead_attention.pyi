"""
This type stub file was generated by pyright.
"""

import torch.nn as nn

class SingleHeadAttention(nn.Module):
    """
    Single-head attention that supports Gating and Downsampling
    """
    def __init__(self, out_channels, embed_dim, head_dim, head_index, dropout=..., bias=..., project_input=..., gated=..., downsample=..., num_heads=...) -> None:
        ...
    
    def forward(self, query, key, value, mask_future_timesteps=..., key_padding_mask=..., use_scalar_bias=...): # -> tuple[Any, Any]:
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        ...
    


class DownsampledMultiHeadAttention(nn.ModuleList):
    """
    Multi-headed attention with Gating and Downsampling
    """
    def __init__(self, out_channels, embed_dim, num_heads, dropout=..., bias=..., project_input=..., gated=..., downsample=...) -> None:
        ...
    
    def forward(self, query, key, value, mask_future_timesteps=..., key_padding_mask=..., use_scalar_bias=...): # -> tuple[Any, Any] | tuple[Tensor, Tensor]:
        ...
    


class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """
    def __init__(self, index) -> None:
        ...
    
    def forward(self, x):
        ...
    


def Linear(in_features, out_features, dropout=..., bias=...): # -> Linear:
    """Weight-normalized Linear layer (input: B x T x C)"""
    ...

def GatedLinear(in_features, out_features, dropout=..., bias=...): # -> Sequential:
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    ...

