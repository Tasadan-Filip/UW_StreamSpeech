# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq.models.transformer import Linear

from ctc_unity.modules.transformer_decoder_base import TransformerDecoderBase
from fairseq.models.transformer import TransformerConfig
from fairseq import utils

logger = logging.getLogger(__name__)


class UnitCTCDecoder(TransformerDecoderBase):
    """
    Unit CTC Decoder

    Based on Transformer decoder, with support to decoding stacked units
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

        self.ctc_upsample_rate = args.ctc_upsample_rate

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        streaming_config=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self._extract_features(
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            streaming_config=streaming_config,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self._output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

    def _output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def _extract_features(
        self,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        streaming_config=None,
    ):

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        slen, bs, embed = enc.size()
        x = (
            enc.unsqueeze(1)
            .repeat(1, self.ctc_upsample_rate, 1, 1)
            .contiguous()
            .view(slen * self.ctc_upsample_rate, bs, embed)
        )
        _x = x.contiguous()

        prev_key_length = 0
        if (
            incremental_state is not None
            and self.layers[0].self_attn._get_input_buffer(incremental_state) != {}
        ):
            prev_key_length = (
                self.layers[0]
                .self_attn._get_input_buffer(incremental_state)["prev_key"]
                .size(-2)
            )

            if x.size(0) > prev_key_length:
                x = x[prev_key_length:]

        if self.embed_positions is not None:
            positions = self.embed_positions(
                x[:, :, 0], incremental_state=incremental_state
            )

        x += positions
        x = self.dropout_module(x)

        self_attn_padding_mask: Optional[Tensor] = None

        if padding_mask is not None and (
            self.cross_self_attention or padding_mask.any()
        ):
            self_attn_padding_mask = (
                padding_mask.unsqueeze(2)
                .repeat(1, 1, self.ctc_upsample_rate)
                .contiguous()
                .view(bs, slen * self.ctc_upsample_rate)
            )

        if streaming_config is not None:
            if (
                "streaming_mask" in streaming_config.keys()
                and streaming_config["streaming_mask"] is not None
            ):
                streaming_mask = streaming_config["streaming_mask"]
                streaming_mask = streaming_mask[:, prev_key_length:]
            else:
                streaming_mask = self._build_streaming_mask(
                    x,
                    enc.size(0),
                    _x.size(0),
                    streaming_config["src_wait"],
                    streaming_config["src_step"],
                    streaming_config["src_step"] * self.ctc_upsample_rate,
                )
                streaming_mask = streaming_mask[prev_key_length:]

        else:
            streaming_mask = None

        # decoder layers
        attention_result_vector: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):

            self_attn_mask = self._buffered_future_mask(_x)
            self_attn_mask = self_attn_mask[-1 * x.size(0) :]

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                extra={"streaming_mask": streaming_mask},
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attention_result_vector = layer_attn.float().to(x)

        if attention_result_vector is not None:
            if alignment_heads is not None:
                attention_result_vector = attention_result_vector[:alignment_heads]

            # average probabilities over heads
            attention_result_vector = attention_result_vector.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {
            "attn": [attention_result_vector],
            "inner_states": inner_states,
            "decoder_padding_mask": self_attn_padding_mask,
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

    def _build_streaming_mask(self, x, src_len, tgt_len, src_wait, src_step, tgt_step):
        idx = torch.arange(0, tgt_len, device=x.device).unsqueeze(1)
        idx = (idx // tgt_step + 1) * src_step + src_wait
        idx = idx.clamp(1, src_len)
        tmp = torch.arange(0, src_len, device=x.device).unsqueeze(0).repeat(tgt_len, 1)
        return tmp >= idx

    def _upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]
