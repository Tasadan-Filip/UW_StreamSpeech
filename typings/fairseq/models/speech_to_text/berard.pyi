"""
This type stub file was generated by pyright.
"""

import torch.nn as nn
from typing import List, Tuple
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel, FairseqIncrementalDecoder, register_model, register_model_architecture

@register_model("s2t_berard")
class BerardModel(FairseqEncoderDecoderModel):
    """Implementation of a model similar to https://arxiv.org/abs/1802.04200

    Paper title: End-to-End Automatic Speech Translation of Audiobooks
    An implementation is available in tensorflow at
    https://github.com/eske/seq2seq
    Relevant files in this implementation are the config
    (https://github.com/eske/seq2seq/blob/master/config/LibriSpeech/AST.yaml)
    and the model code
    (https://github.com/eske/seq2seq/blob/master/translate/models.py).
    The encoder and decoder try to be close to the original implementation.
    The attention is an MLP as in Bahdanau et al.
    (https://arxiv.org/abs/1409.0473).
    There is no state initialization by averaging the encoder outputs.
    """
    def __init__(self, encoder, decoder) -> None:
        ...
    
    @staticmethod
    def add_args(parser): # -> None:
        ...
    
    @classmethod
    def build_encoder(cls, args, task): # -> FairseqEncoder | FairseqDecoder | BerardEncoder:
        ...
    
    @classmethod
    def build_decoder(cls, args, task): # -> FairseqEncoder | FairseqDecoder | LSTMDecoder:
        ...
    
    @classmethod
    def build_model(cls, args, task): # -> Self:
        """Build a new model instance."""
        ...
    
    def get_normalized_probs(self, net_output, log_probs, sample=...): # -> Any | Tensor:
        ...
    


class BerardEncoder(FairseqEncoder):
    def __init__(self, input_layers: List[int], conv_layers: List[Tuple[int]], in_channels: int, input_feat_per_channel: int, num_blstm_layers: int, lstm_size: int, dropout: float) -> None:
        """
        Args:
            input_layers: list of linear layer dimensions. These layers are
                applied to the input features and are followed by tanh and
                possibly dropout.
            conv_layers: list of conv2d layer configurations. A configuration is
                a tuple (out_channels, conv_kernel_size, stride).
            in_channels: number of input channels.
            input_feat_per_channel: number of input features per channel. These
                are speech features, typically 40 or 80.
            num_blstm_layers: number of bidirectional LSTM layers.
            lstm_size: size of the LSTM hidden (and cell) size.
            dropout: dropout probability. Dropout can be applied after the
                linear layers and LSTM layers but not to the convolutional
                layers.
        """
        ...
    
    def forward(self, src_tokens, src_lengths=..., **kwargs): # -> dict[str, Any | Tensor]:
        """
        Args
            src_tokens: padded tensor (B, T, C * feat)
            src_lengths: tensor of original lengths of input utterances (B,)
        """
        ...
    
    def reorder_encoder_out(self, encoder_out, new_order):
        ...
    


class MLPAttention(nn.Module):
    """The original attention from Badhanau et al. (2014)

    https://arxiv.org/abs/1409.0473, based on a Multi-Layer Perceptron.
    The attention score between position i in the encoder and position j in the
    decoder is: alpha_ij = V_a * tanh(W_ae * enc_i + W_ad * dec_j + b_a)
    """
    def __init__(self, decoder_hidden_state_dim, context_dim, attention_dim) -> None:
        ...
    
    def forward(self, decoder_state, source_hids, encoder_padding_mask): # -> tuple[Any, Tensor]:
        """The expected input dimensions are:
        decoder_state: bsz x decoder_hidden_state_dim
        source_hids: src_len x bsz x context_dim
        encoder_padding_mask: src_len x bsz
        """
        ...
    


class LSTMDecoder(FairseqIncrementalDecoder):
    def __init__(self, dictionary, embed_dim, num_layers, hidden_size, dropout, encoder_output_dim, attention_dim, output_layer_dim) -> None:
        """
        Args:
            dictionary: target text dictionary.
            embed_dim: embedding dimension for target tokens.
            num_layers: number of LSTM layers.
            hidden_size: hidden size for LSTM layers.
            dropout: dropout probability. Dropout can be applied to the
                embeddings, the LSTM layers, and the context vector.
            encoder_output_dim: encoder output dimension (hidden size of
                encoder LSTM).
            attention_dim: attention dimension for MLP attention.
            output_layer_dim: size of the linear layer prior to output
                projection.
        """
        ...
    
    def forward(self, prev_output_tokens, encoder_out=..., incremental_state=..., **kwargs): # -> tuple[Any, None]:
        ...
    
    def reorder_incremental_state(self, incremental_state, new_order): # -> None:
        ...
    


@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard")
def berard(args): # -> None:
    """The original version: "End-to-End Automatic Speech Translation of
    Audiobooks" (https://arxiv.org/abs/1802.04200)
    """
    ...

@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_256_3_3")
def berard_256_3_3(args): # -> None:
    """Used in
    * "Harnessing Indirect Training Data for End-to-End Automatic Speech
    Translation: Tricks of the Trade" (https://arxiv.org/abs/1909.06515)
    * "CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus"
    (https://arxiv.org/pdf/2002.01320.pdf)
    * "Self-Supervised Representations Improve End-to-End Speech Translation"
    (https://arxiv.org/abs/2006.12124)
    """
    ...

@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_512_3_2")
def berard_512_3_2(args): # -> None:
    ...

@register_model_architecture(model_name="s2t_berard", arch_name="s2t_berard_512_5_3")
def berard_512_5_3(args): # -> None:
    ...

