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


class CTCDecoder(nn.Module):
    def __init__(self, tgt_dict, models):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.unk = tgt_dict.unk()
        self.models = models
        self.tgt_dict = tgt_dict

    @torch.no_grad()
    def generate(self, encoder_out, prefix=None, aux_task_name=None, **kwargs):
        model = self.models[0]
        model.eval()

        max_len = model.max_decoder_positions()
        # TODO: incorporate max_len_a and max_len_b

        incremental_state = {}
        pred_out, attn, scores = [], [], []

        prev_output_tokens = None
        decoder_name = f"{aux_task_name}_decoder" if aux_task_name else "decoder"
        ctc_decoder = getattr(model, decoder_name)
        ctc_out = ctc_decoder(encoder_out["encoder_out"][0], **kwargs)
        lprobs = model.get_normalized_probs(
            [ctc_out["encoder_out"].transpose(0, 1)], log_probs=True
        )
        # never select pad, unk
        lprobs[:, :, self.pad] = -math.inf
        lprobs[:, :, self.unk] = -math.inf

        cur_pred_lprob, cur_pred_out = torch.max(lprobs, dim=2)
        scores = cur_pred_lprob
        pred_out = cur_pred_out
        attn = None
        alignment = None

        def _ctc_postprocess(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [
                v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]
            ]
            hyp = [
                v
                for v in deduplicated_toks
                if (v != 0) and (v != self.tgt_dict.pad_index)
            ]
            return torch.tensor(hyp)

        def _ctc_postprocess_index(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [
                (v, i) for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]
            ]
            index = [
                i
                for v, i in deduplicated_toks
                if (v != 0) and (v != self.tgt_dict.pad_index)
            ]
            return index

        if prefix is not None:

            pred_out = torch.cat((prefix, pred_out[:, prefix.size(1) :]), dim=1)

        hypos = [
            [
                {
                    "tokens": _ctc_postprocess(pred_out[b]),
                    "org_tokens": pred_out[b],
                    "lprobs": lprobs,
                    "index": _ctc_postprocess_index(pred_out[b]),
                    "attn": None,
                    "alignment": None,
                    "positional_scores": scores[b],
                    "score": utils.item(scores[b].sum().data),
                }
            ]
            for b in range(pred_out.size(0))
        ]

        return hypos
