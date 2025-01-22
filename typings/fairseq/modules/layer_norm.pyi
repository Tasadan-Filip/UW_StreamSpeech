"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm as _FusedLayerNorm

has_fused_layernorm = ...
class FusedLayerNorm(_FusedLayerNorm):
    @torch.jit.unused
    def forward(self, x):
        ...
    


def LayerNorm(normalized_shape, eps=..., elementwise_affine=..., export=...): # -> FusedLayerNorm | LayerNorm:
    ...

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def forward(self, input): # -> Tensor:
        ...
    


