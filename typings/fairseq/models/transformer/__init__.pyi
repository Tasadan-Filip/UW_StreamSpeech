"""
This type stub file was generated by pyright.
"""

from .transformer_config import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, DEFAULT_MIN_PARAMS_TO_WRAP, TransformerConfig
from .transformer_decoder import Linear, TransformerDecoder, TransformerDecoderBase
from .transformer_encoder import TransformerEncoder, TransformerEncoderBase
from .transformer_legacy import TransformerModel, base_architecture, tiny_architecture, transformer_iwslt_de_en, transformer_vaswani_wmt_en_de_big, transformer_vaswani_wmt_en_fr_big, transformer_wmt_en_de, transformer_wmt_en_de_big, transformer_wmt_en_de_big_t2t
from .transformer_base import Embedding, TransformerModelBase

"""isort:skip_file"""
__all__ = ["TransformerModelBase", "TransformerConfig", "TransformerDecoder", "TransformerDecoderBase", "TransformerEncoder", "TransformerEncoderBase", "TransformerModel", "Embedding", "Linear", "base_architecture", "tiny_architecture", "transformer_iwslt_de_en", "transformer_wmt_en_de", "transformer_vaswani_wmt_en_de_big", "transformer_vaswani_wmt_en_fr_big", "transformer_wmt_en_de_big", "transformer_wmt_en_de_big_t2t", "DEFAULT_MAX_SOURCE_POSITIONS", "DEFAULT_MAX_TARGET_POSITIONS", "DEFAULT_MIN_PARAMS_TO_WRAP"]
