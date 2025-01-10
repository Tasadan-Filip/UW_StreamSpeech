import torch
from typing import Optional
from ctc_unity.modules.streaming_speech_encoder.streaming_speech_encoder_modules.streaming_speech_encoder_convolution_layer import StreamingSpeechEncoderConvolutionLayer
from ctc_unity.modules.streaming_speech_encoder.streaming_speech_encoder_modules.streaming_speech_encoder_feed_forward_network import StreamingSpeechEncoderFeedForwardNetwork
from fairseq.modules import LayerNorm
from uni_unity.modules.espnet_multihead_attention import (
    RelPositionMultiHeadedAttention
)

class StreamingSpeechEncoderLayer(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100. We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
        chunk_size=None,
    ):
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
        self.pos_enc_type = pos_enc_type
        super(StreamingSpeechEncoderLayer, self).__init__()

        self.ffn1 = StreamingSpeechEncoderFeedForwardNetwork(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = LayerNorm(embed_dim, export=False)

        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                    dropout=dropout
                )  
             
        self.convolution_layer = StreamingSpeechEncoderConvolutionLayer(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation_fn=activation_fn,
            chunk_size=chunk_size,
        )

        self.ffn2 = StreamingSpeechEncoderFeedForwardNetwork(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = LayerNorm(embed_dim, export=False)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
        extra=None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                pos_emb=position_emb,
                need_weights=False,
                extra=extra,
            )

        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.convolution_layer(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        layer_result = x

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x, (attn, layer_result)
