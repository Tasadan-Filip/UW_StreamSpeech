"""
This type stub file was generated by pyright.
"""

import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=..., precision=...) -> None:
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        ...
    
    def forward(self, x, seq_len=...): # -> tuple[Tensor | None, Tensor | None]:
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        ...
    


def rotate_half(x): # -> Tensor:
    ...

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = ...): # -> tuple[Any, Any]:
    ...

