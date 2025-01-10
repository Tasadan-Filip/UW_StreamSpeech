# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.modules import LayerNorm
from ctc_unity.modules.text_to_unit_encoder.text_to_unit_encoder_layer import TextToUnitEncoderLayer
from fairseq.models.transformer import TransformerConfig

class TextToUnitEncoder(FairseqEncoder):
    """Transformer encoder without token embeddings."""

    def __init__(self, args):
        super().__init__(None)

        self.layers = nn.ModuleList(
            [TextToUnitEncoderLayer(TransformerConfig.from_namespace(args)) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self._future_mask = torch.empty(0)
        self.unidirectional = getattr(args, "uni_encoder", False)

    def forward(
        self, x, encoder_padding_mask, return_all_hiddens=False, streaming_config=None
    ):
        encoder_states = []

        extra = {
                "encoder_mask": (
                    self._buffered_future_mask(x) if self.unidirectional else None
                )
            }

        for layer in self.layers:
            x = layer(x, encoder_padding_mask, extra=extra)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": (
                [encoder_padding_mask]
                if encoder_padding_mask is not None and encoder_padding_mask.any()
                else []
            ),  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def _buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]