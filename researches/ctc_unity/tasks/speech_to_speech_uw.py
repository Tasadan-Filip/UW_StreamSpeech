from pathlib import Path
import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask

logger = logging.getLogger(__name__)

@register_task("speech_to_speech_uw")
class UWSpeechToSpeechTask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        # ./fr-en/config_gcmvn.yaml
        parser.add_argument(     
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        # ./fr-en/config_mtl_asr_st_ctcst.yaml
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Configuration YAML filename for the multitasks (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--target-is-code",
            action="store_true",
            help="set if target is discrete unit instead of spectrogram",
        )
        parser.add_argument(
            "--target-code-size", type=int, default=None, help="# discrete units"
        )
        parser.add_argument(
            "--n-frames-per-step",
            type=int,
            default=1,
            help="# stacked frames, use 0 for reduced discrete unit sequence",
        )
        parser.add_argument("--eval-inference", action="store_true")
        parser.add_argument(
            "--eval-args",
            type=str,
            default="{}",
            help='generation args for speech-to-unit model , e.g., \'{"beam": 5, "max_len_a": 1}\', as JSON string',
        )
        parser.add_argument("--eos-prob-threshold", type=float, default=0.5)
        parser.add_argument(
            "--mcd-normalize-type",
            type=str,
            default="targ",
            choices=["targ", "pred", "path"],
        )
        parser.add_argument(
            "--vocoder",
            type=str,
            default="griffin_lim",
            choices=["griffin_lim", "hifigan", "code_hifigan"],
        )
        parser.add_argument("--spec-bwd-max-iter", type=int, default=8)
        parser.add_argument(
            "--infer-target-lang",
            type=str,
            default="",
            help="target language for inference",
        )

    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        # insert blank symbols
        tgt_blank_index = tgt_dict.add_symbol("<blank>")
        self.tgt_dict = tgt_dict
        self.tgt_dict.blank_index = tgt_blank_index

        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)

        self.multitask_tasks = {}
        self.tgt_dict_mt = None
        self.eos_token_mt = None 

        ## setup multitask task config
        if getattr(args, "multitask_config_yaml", None) is not None:
            multitask_cfg = MultitaskConfig(
                Path(args.data) / args.multitask_config_yaml
            )
            first_pass_task_idx = multitask_cfg.first_pass_decoder_task_index
            for i, (task_name, task_config) in enumerate(
                multitask_cfg.get_all_tasks().items()
            ):
                task_obj = DummyMultiTask(
                    task_config,
                    task_config.tgt_dict,
                    first_pass=i == first_pass_task_idx,
                )
                self.multitask_tasks[task_name] = task_obj
                if task_obj.is_first_pass_decoder:
                    self.tgt_dict_mt = task_obj.target_dictionary
                    if task_config.prepend_bos_and_append_tgt_lang_tag:
                        self.eos_token_mt = task_config.eos_token
                        assert not isinstance(self.eos_token_mt, List)

                        if not self.eos_token_mt:
                            raise Warning(
                                "Please provide eos_token in --multitask-config-yaml to replace eos in sequence generator"
                            )

        self._infer_tgt_lang_id = infer_tgt_lang_id
        self.blank_symbol = "<blank>"

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict = None
        infer_tgt_lang_id = None

        ## setup the target dictionary
        if args.target_is_code: # true in config
            if data_cfg.prepend_tgt_lang_tag_as_bos:
                # dictionary with language tags
                dict_path = Path(args.data) / data_cfg.vocab_filename
                if not dict_path.is_file():
                    raise FileNotFoundError(
                        f"Dict has to be provided when setting prepend_tgt_lang_tag_as_bos: true, but dict not found: {dict_path}"
                    )
                tgt_dict = Dictionary.load(dict_path.as_posix())

                # target langauge for inference
                if args.infer_target_lang != "":
                    tgt_lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(
                        args.infer_target_lang
                    )
                    infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
                    assert infer_tgt_lang_id != tgt_dict.unk()
            else:
                assert args.target_code_size is not None

                tgt_dict = Dictionary()
                for i in range(args.target_code_size):
                    tgt_dict.add_symbol(str(i))
            logger.info(f"dictionary size: " f"{len(tgt_dict):,}")

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        assert args.n_frames_per_step >= 1
        assert (
            not args.eval_inference
            or (args.target_is_code and args.vocoder == "code_hifigan")
            or (not args.target_is_code and args.vocoder != "code_hifigan")
        )

        # call __init__ functions
        return cls(args, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)

    def build_criterion(self, args):
        from fairseq import criterions

        if len(self.multitask_tasks) > 0:
            if self.args.target_is_code and not args._name.startswith("speech_to_unit"):
                raise ValueError(
                    "set --criterion speech_to_unit for speech-to-unit loss with multitask"
                )
            elif not self.args.target_is_code and not args._name.startswith(
                "speech_to_spectrogram"
            ):
                raise ValueError(
                    "set --criterion speech_to_spectrogram for speech-to-spectrogram loss with multitask"
                )

        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = SpeechToSpeechDatasetCreator.from_tsv(
            root=self.args.data,
            data_cfg=self.data_cfg,
            splits=split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def target_dictionary_mt(self):
        return self.tgt_dict_mt

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args, from_checkpoint=False):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_transformed_channels
        args.target_speaker_embed = self.data_cfg.target_speaker_embed is not None
        args.n_frames_per_step = self.args.n_frames_per_step

        model = super().build_model(args, from_checkpoint)

        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        for task_name, task_obj in self.multitask_tasks.items():
            criterion.set_multitask_loss_weight(
                task_name, task_obj.args.get_loss_weight(update_num)
            )
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].train()

        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        for task_name in self.multitask_tasks.keys():
            if task_name in model.multitask_decoders:
                model.multitask_decoders[task_name].eval()
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        return loss, sample_size, logging_output