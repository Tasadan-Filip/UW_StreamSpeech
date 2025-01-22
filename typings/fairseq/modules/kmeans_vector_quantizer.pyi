"""
This type stub file was generated by pyright.
"""

import torch.nn as nn

class KmeansVectorQuantizer(nn.Module):
    def __init__(self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=...) -> None:
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        ...
    
    @property
    def expand_embedding(self): # -> Tensor | Parameter:
        ...
    
    def forward_idx(self, x): # -> tuple[Any, Any]:
        ...
    
    def forward(self, x, produce_targets=...): # -> dict[str, Any]:
        ...
    


