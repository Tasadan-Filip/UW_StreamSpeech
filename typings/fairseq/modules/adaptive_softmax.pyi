"""
This type stub file was generated by pyright.
"""

from torch import nn

class TiedLinear(nn.Module):
    def __init__(self, weight, transpose) -> None:
        ...
    
    def forward(self, input): # -> Tensor:
        ...
    


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, num_classes, q_noise, qn_block_size) -> None:
        ...
    
    def forward(self, input): # -> Any | Tensor:
        ...
    


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """
    def __init__(self, vocab_size, input_dim, cutoff, dropout, factor=..., adaptive_inputs=..., tie_proj=..., q_noise=..., qn_block_size=...) -> None:
        ...
    
    def upgrade_state_dict_named(self, state_dict, name): # -> None:
        ...
    
    def adapt_target(self, target): # -> tuple[list[Any], list[Any]]:
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """
        ...
    
    def forward(self, input, target): # -> tuple[list[Any], list[Any]]:
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """
        ...
    
    def get_log_prob(self, input, target): # -> Any:
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """
        ...
    


