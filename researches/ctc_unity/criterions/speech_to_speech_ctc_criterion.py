# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import torch
from dataclasses import dataclass, field
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
)
import torch.nn.functional as F
from fairseq.criterions.speech_to_speech_criterion import (
    SpeechToUnit2passMultitaskTaskCriterion,
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechToUnit2passCTCCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    k1: int = field(
        default=3,
        metadata={"help": "k1"},
    )
    k2: int = field(
        default=3,
        metadata={"help": "k2"},
    )
    n1: int = field(
        default=3,
        metadata={"help": "n1"},
    )
    n2: int = field(
        default=3,
        metadata={"help": "n2"},
    )
    unit_per_subword: int = field(
        default=10,
        metadata={"help": "k1"},
    )
    segment_size: int = field(
        default=280,
        metadata={"help": "k1"},
    )
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    multichunk: bool = field(
        default=False,
        metadata={"help": "multi_chunk"},
    )


@register_criterion(
    "speech_to_unit_2pass_ctc", dataclass=SpeechToUnit2passCTCCriterionConfig
)
class SpeechToUnit2passCTCMultitaskTaskCriterion(
    SpeechToUnit2passMultitaskTaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
        k1=3,
        k2=1,
        n1=3,
        n2=3,
        unit_per_subword=10,
        segment_size=280,
        post_process="letter",
        multichunk=False,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            rdrop_alpha,
        )
        self.k1 = k1
        self.k2 = k2
        self.n1 = n1
        self.n2 = n2
        self.unit_per_subword = unit_per_subword
        self.segment_size = segment_size
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = post_process

        self.multichunk = multichunk

    def forward(self, model, sample, reduce=True):
        net_input_concat = {
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "prev_output_tokens_mt": sample["multitask"][model.mt_task_name][
                "net_input"
            ]["prev_output_tokens"],
            "tgt_speaker": sample["net_input"].get("tgt_speaker", None),
            "return_all_hiddens": True,
        }

        # Handle optional ASR task-specific inputs
        if getattr(model, "asr_task_name", None) is not None:
            net_input_concat["prev_output_tokens_asr"] = sample["multitask"][
                model.asr_task_name
            ]["net_input"]["prev_output_tokens"]

        # Configure chunk size dynamically for multi-chunk processing
        num_updates = model.encoder.num_updates
        if self.multichunk:
            if not model.training:
                chunk_size = 99999
            else:
                chunk_size = random.choice([8, 16, 24, 32, 99999])
            chunk_size = int(chunk_size)

            model.encoder.chunk_size = chunk_size

            if not model.training and num_updates < 20000:
                conv_chunk_size = 8
            else:
                conv_chunk_size = random.choice([8, 16])

            chunk_size = min(chunk_size, conv_chunk_size)

            for conv in model.encoder.subsample.conv_layers:
                conv.chunk_size = chunk_size
            for layer in model.encoder.conformer_layers:
                layer.conv_module.depthwise_conv.chunk_size = chunk_size

        # Obtain model outputs
        net_output, extra = model(**net_input_concat)

        # Compute the loss and negative log-likelihood loss
        # NOTE: seems like loss and nll_loss are always the same?
        loss, nll_loss = self.compute_loss(
            model, [net_output, extra], sample, reduce=reduce
        )

        # Determine sample size for loss averaging
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # Compute accuracy if required and model is in evaluation mode
        if self.report_accuracy and not model.training:
            n_correct, total = self.compute_accuracy(model, [net_output, extra], sample)
            logging_output["n_correct"] = n_correct
            logging_output["total"] = total

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # Add multitask losses and logs if multitask
        # NOTE: get_multitask_loss is from MultitaskCriterion, migrating it seems overkill
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        # Compute log probabilities from the model's output
        lprobs = model.get_normalized_probs(net_output, log_probs=True).transpose(0, 1)

        # Get the target labels
        target = model.get_targets(sample, net_output)

        # Compute lengths for padding masks
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        if net_output[-1]["decoder_padding_mask"] is not None:
            non_padding_mask = ~net_output[-1]["decoder_padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(0),), lprobs.size(1), dtype=torch.long
            )

        # Compute the CTC loss
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                target,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )

        return loss, loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True).transpose(0, 1)

        if net_output[-1]["decoder_padding_mask"] is not None:
            non_padding_mask = ~net_output[-1]["decoder_padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(0),), lprobs.size(1), dtype=torch.long
            )

        logging_output = {}
        import editdistance

        with torch.no_grad():
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

            # Compute edit distance for accuracy
            c_err = 0
            c_len = 0
            for lp, t, inp_l in zip(
                lprobs_t,
                (
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"]
                ),
                input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)

                # Filter out padding and EOS tokens from the target
                p = (t != self.task.target_dictionary.pad()) & (
                    t != self.task.target_dictionary.eos()
                )
                targ = t[p]
                targ_units_arr = targ.tolist()

                # Get predictions and compute edit distance
                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len
        return (
            logging_output["c_total"] - logging_output["c_errors"],
            logging_output["c_total"],
        )


