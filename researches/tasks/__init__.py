from argparse import Namespace
import dataclasses
from re import A
from typing import Literal, Protocol
from uu import decode
from fairseq.models import BaseFairseqModel
from fairseq.tasks.fairseq_task import LegacyFairseqTask
import torch


class DummyMultiTask(LegacyFairseqTask):
    @dataclasses.dataclass
    class Args(Namespace):
        input_from: str
        decoder_type: Literal["ctc"]

    args: Args

    def __init__(self, args: Args, tgt_dict, first_pass=False):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.first_pass = first_pass

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def is_first_pass_decoder(self):
        return self.first_pass

    def inference_step(
        self,
        generator,
        models: list[BaseFairseqModel],
        sample,
        prefix_tokens=None,
        constraints=None,
    ):
        if self.args.decoder_type == "ctc":
            model = models[0]  # only support single model
            encoder_out = model(**sample)
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(
                    encoder_out
                )  # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
            return generator.decode(
                emissions.transpose(0, 1).float().cpu().contiguous()
            )
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")

    def build_generator(
        self,
        models,
        args: Args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        if self.args.decoder_type == "ctc":
            from fairseq.examples.speech_recognition.w2l_decoder import (
                W2lViterbiDecoder,
            )

            return W2lViterbiDecoder(args, self.tgt_dict)
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")
