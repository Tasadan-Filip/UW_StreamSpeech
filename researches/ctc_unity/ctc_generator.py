# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
from fairseq import utils

logger = logging.getLogger(__name__)


class CTCSequenceGenerator(nn.Module):
    def __init__(self, tgt_dict, models):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.unk = tgt_dict.unk()
        self.models = models
        self.tgt_dict = tgt_dict

    @torch.no_grad()
    def generate(self, encoder_out, **kwargs):
        # currently only support viterbi search for stacked units
        model = self.models[0]
        model.eval()

        max_len = model.max_decoder_positions()
        # TODO: incorporate max_len_a and max_len_b

        incremental_state = {}
        pred_out, attn, scores = [], [], []

        prev_output_tokens = None
        ctc_decoder = model.decoder
        ctc_out, ctc_extra = ctc_decoder(None, encoder_out=encoder_out)
        lprobs = model.get_normalized_probs([ctc_out], log_probs=True)
        # never select pad, unk
        lprobs[:, :, self.pad] = -math.inf
        lprobs[:, :, self.unk] = -math.inf
        lprobs[:, :, self.eos] = -math.inf

        cur_pred_lprob, cur_pred_out = torch.max(lprobs, dim=2)
        scores = cur_pred_lprob
        pred_out = cur_pred_out
        attn = ctc_extra["attn"][0]
        alignment = None

        def _ctc_postprocess(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [
                v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]
            ]
            hyp = [
                v
                for v in deduplicated_toks
                if (v != self.tgt_dict.blank_index) and (v != self.tgt_dict.pad_index)
            ]
            return torch.tensor(hyp)

        hypos = [
            [
                {
                    "tokens": _ctc_postprocess(pred_out[b]),
                    "attn": None,
                    "alignment": None,
                    "positional_scores": scores[b],
                    "score": utils.item(scores[b].sum().data),
                }
            ]
            for b in range(pred_out.size(0))
        ]

        return hypos
