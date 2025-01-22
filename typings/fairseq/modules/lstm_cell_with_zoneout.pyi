"""
This type stub file was generated by pyright.
"""

import torch.nn as nn

class LSTMCellWithZoneOut(nn.Module):
    """
    Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations
    https://arxiv.org/abs/1606.01305
    """
    def __init__(self, prob: float, input_size: int, hidden_size: int, bias: bool = ...) -> None:
        ...
    
    def zoneout(self, h, next_h, prob): # -> tuple[tuple[Any, ...] | Any, ...]:
        ...
    
    def forward(self, x, h): # -> tuple[tuple[Any, ...] | Any, ...]:
        ...
    


