"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional

class ConvolutionModule(torch.nn.Module):
    """Convolution block used in the conformer block"""
    def __init__(self, embed_dim, channels, depthwise_kernel_size, dropout, activation_fn=..., bias=..., export=...) -> None:
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            dropout: dropout value
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
            export: If layernorm should be exported to jit
        """
        ...
    
    def forward(self, x): # -> Any:
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        ...
    


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""
    def __init__(self, input_feat, hidden_units, dropout1, dropout2, activation_fn=..., bias=...) -> None:
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            dropout1: dropout value for layer1
            dropout2: dropout value for layer2
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """
        ...
    
    def forward(self, x): # -> Any:
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        ...
    


class ConformerEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""
    def __init__(self, embed_dim, ffn_embed_dim, attention_heads, dropout, use_fp16, depthwise_conv_kernel_size=..., activation_fn=..., attn_type=..., pos_enc_type=...) -> None:
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            dropout: dropout value
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation from ESPNET vs fairseq
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        ...
    
    def forward(self, x, encoder_padding_mask: Optional[torch.Tensor], position_emb: Optional[torch.Tensor] = ...): # -> tuple[Any, tuple[Any, Any]]:
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        ...
    


class ConformerWav2Vec2EncoderLayer(ConformerEncoderLayer):
    """Encoder layer for Wav2vec2 encoder"""
    def forward(self, x: torch.Tensor, self_attn_mask: torch.Tensor = ..., self_attn_padding_mask: torch.Tensor = ..., need_weights: bool = ..., att_args=..., position_emb=...): # -> tuple[Any, tuple[Any, Any]]:
        ...
    


