"""
This type stub file was generated by pyright.
"""

import torch.nn as nn

class LocationAttention(nn.Module):
    """
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf

    :param int encoder_dim: # projection-units of encoder
    :param int decoder_dim: # units of decoder
    :param int attn_dim: attention dimension
    :param int conv_dim: # channels of attention convolution
    :param int conv_kernel_size: filter size of attention convolution
    """
    def __init__(self, attn_dim, encoder_dim, decoder_dim, attn_state_kernel_size, conv_dim, conv_kernel_size, scaling=...) -> None:
        ...
    
    def clear_cache(self): # -> None:
        ...
    
    def forward(self, encoder_out, encoder_padding_mask, decoder_h, attn_state): # -> tuple[Tensor, Tensor]:
        """
        :param torch.Tensor encoder_out: padded encoder hidden state B x T x D
        :param torch.Tensor encoder_padding_mask: encoder padding mask
        :param torch.Tensor decoder_h: decoder hidden state B x D
        :param torch.Tensor attn_prev: previous attention weight B x K x T
        :return: attention weighted encoder state (B, D)
        :rtype: torch.Tensor
        :return: previous attention weights (B x T)
        :rtype: torch.Tensor
        """
        ...
    


