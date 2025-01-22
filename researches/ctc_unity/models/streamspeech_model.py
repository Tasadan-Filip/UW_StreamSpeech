# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from pathlib import Path
from ctc_unity.modules.simultaneous_text_decoder import SimultaneousTextDecoder
from ctc_unity.modules.streaming_speech_encoder.streaming_speech_encoder import StreamingSpeechEncoder
import torch
from typing import OrderedDict

from fairseq.models import (
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.modules import CTCDecoder
from fairseq.models.speech_to_speech.modules import StackedEmbedding
from ctc_unity.modules.text_to_unit_encoder.text_to_unit_encoder import (
    TextToUnitEncoder
)
from fairseq.models.speech_to_speech.s2s_transformer import (
    base_multitask_text_transformer_decoder_arch,
    s2ut_architecture_base,
)
from fairseq.models.transformer import TransformerModelBase
from ctc_unity.modules.unit_ctc_decoder import UnitCTCDecoder

from fairseq import checkpoint_utils, utils
from fairseq.models.transformer import TransformerConfig

from fairseq.models import FairseqEncoderDecoderModel

logger = logging.getLogger(__name__)

@register_model("streamspeech")
class StreamSpeechModel(FairseqEncoderDecoderModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + Transformer discrete unit decoder
    """

    @classmethod
    def build_model(cls, args, task):
        streaming_speech_encoder = cls._build_streaming_speech_encoder(args)
        unit_ctc_decoder = cls._build_unit_ctc_decoder(
            args,
            task.target_dictionary
        )
        base_model = cls(streaming_speech_encoder, unit_ctc_decoder)

        base_model.t2u_augmented_cross_attn = getattr(
            args, "synthesizer_augmented_cross_attention", False
        )

        # set up multitask decoders
        base_model.mt_task_name = None
        base_model.multitask_decoders = {}
        has_first_pass_decoder = False
        for task_name, task_obj in task.multitask_tasks.items():
            if task_obj.is_first_pass_decoder:
                has_first_pass_decoder = True
                base_model.mt_task_name = task_name

            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )

            task_decoder = cls._build_multitask_decoder(
                task_obj.args,
                task_obj.target_dictionary,
                in_dim,
                getattr(args, "translation_decoder_layers", 4),
                getattr(args, "decoder_embed_dim", 256),
                getattr(args, "decoder_attention_heads", 4),
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)

            if task_obj.args.decoder_type == "ctc":
                decoder_model_cls = FairseqEncoderModel
            else:
                decoder_model_cls = FairseqLanguageModel
            
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        assert has_first_pass_decoder, "set at least one intermediate non-CTC decoder"

        # set up encoder on top of the auxiliary MT decoder
        if getattr(args, "synthesizer_encoder_layers", 0) > 0:
            #TODO: clean this method
            base_model.synthesizer_encoder = cls._build_text_to_unit_encoder(args)
        else:
            base_model.synthesizer_encoder = None

        if getattr(args, "load_pretrained_mt_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrained_mt_from
            )["model"]
            encoder_state_dict = OrderedDict()
            decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    encoder_state_dict[subkey] = state_dict[key]
                elif key.startswith("decoder"):
                    decoder_state_dict[key] = state_dict[key]
            base_model.encoder.load_state_dict(encoder_state_dict)
            base_model.multitask_decoders[base_model.mt_task_name].load_state_dict(
                decoder_state_dict
            )
            logger.info(
                f"Successfully load pretrained Conformer from {args.load_pretrained_mt_from}."
            )

        return base_model
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_output_tokens_mt,
        streaming_config=None,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):
        mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )

        if streaming_config is not None:
            asr_decoder = getattr(self, "source_unigram_decoder")
            asr_ctc_out = asr_decoder(encoder_out["encoder_out"][0].detach())
            asr_probs = self.get_normalized_probs(
                [asr_ctc_out["encoder_out"].transpose(0, 1)], log_probs=False
            )
            asr_repeat = (
                torch.cat(
                    (
                        torch.zeros(
                            (asr_probs.size(0), 1, asr_probs.size(-1) - 1),
                            device=asr_probs.device,
                        ),
                        asr_probs[:, :-1, 1:],
                    ),
                    dim=1,
                )
                * asr_probs[:, :, 1:]
            )
            asr_repeat = asr_repeat.sum(dim=-1, keepdim=False)
            asr_blank = asr_probs[:, :, 0]
            asr_not_blank = 1 - (asr_repeat + asr_blank).detach()

            st_decoder = getattr(self, "ctc_target_unigram_decoder")
            st_ctc_out = st_decoder(encoder_out["encoder_out"][0].detach())
            st_probs = self.get_normalized_probs(
                [st_ctc_out["encoder_out"].transpose(0, 1)], log_probs=False
            )
            st_repeat = (
                torch.cat(
                    (
                        torch.zeros(
                            (st_probs.size(0), 1, st_probs.size(-1) - 1),
                            device=st_probs.device,
                        ),
                        st_probs[:, :-1, 1:],
                    ),
                    dim=1,
                )
                * st_probs[:, :, 1:]
            )
            st_repeat = st_repeat.sum(dim=-1, keepdim=False)
            st_blank = st_probs[:, :, 0]
            st_not_blank = 1 - (st_repeat + st_blank).detach()

            streaming_mask = self._build_streaming_mask(
                asr_not_blank,
                st_not_blank,
                prev_output_tokens_mt,
                streaming_config["k1"],
                streaming_config["n1"],
                streaming_config["n1"],
            )
            streaming_config["streaming_mask"] = streaming_mask

        # 1. MT decoder
        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=encoder_out,
            streaming_config=streaming_config,
        )
        x = mt_decoder_out[1]["inner_states"][-1]
        if mt_decoder.layer_norm is not None:
            x = mt_decoder.layer_norm(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if self.synthesizer_encoder is not None:
            t2u_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
                return_all_hiddens=return_all_hiddens,
            )
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
            }

        # 3. T2U decoder
        if self.t2u_augmented_cross_attn:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                encoder_out_aug=t2u_encoder_out,
            )
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=t2u_encoder_out,
                streaming_config=(
                    {
                        "src_wait": int(streaming_config["k2"]),
                        "src_step": int(streaming_config["n2"]),
                    }
                    if streaming_config is not None
                    else None
                ),
            )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--translation-decoder-layers",
            type=int,
            default=4,
            metavar="N",
            help="num decoder layers in the first-pass translation module",
        )
        parser.add_argument(
            "--synthesizer",
            default="transformer",
            choices=["transformer"],
            help="",
        )
        parser.add_argument(
            "--synthesizer-encoder-layers",
            type=int,
            default=0,
            metavar="N",
            help="num encoder layers in the second-pass synthesizer module",
        )
        parser.add_argument(
            "--synthesizer-augmented-cross-attention",
            action="store_true",
            default=False,
            help="augmented cross-attention over speech encoder output",
        )
        parser.add_argument(
            "--load-pretrained-mt-from",
            type=str,
            help="path to pretrained s2t transformer model",
        )
        parser.add_argument(
            "--uni-encoder",
            action="store_true",
            default=False,
            help="apply unidirectional encoder",
        )
        parser.add_argument(
            "--ctc-upsample-rate",
            type=int,
            default=10,
            metavar="N",
        )

        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="STR",
            help="kernel sizes of Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv2d (convtransformer) subsampling layers",
        )
        parser.add_argument(
            "--conv-version",
            type=str,
            default="s2t_transformer",
            choices=["s2t_transformer", "convtransformer"],
            help="version of frontend convolutional layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        # speaker
        parser.add_argument(
            "--speaker-embed-dim",
            type=int,
            metavar="N",
            help="speaker embedding dimension",
        )
        
        parser.add_argument(
            "--depthwise-conv-kernel-size",
            type=int,
            metavar="N",
            help="kernel size of depthwise convolution layers",
        )
        parser.add_argument(
            "--attn-type",
            type=str,
            metavar="STR",
            help="If not specified uses fairseq MHA. Other valid option is espnet for using conformer",
        )
        parser.add_argument(
            "--pos-enc-type",
            type=str,
            metavar="STR",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            metavar="N",
            default=-1,
            help="chunk size",
        )

    @classmethod
    def _build_multitask_decoder(
        cls,
        args,
        tgt_dict,
        in_dim,
        decoder_layers,
        decoder_embed_dim,
        decoder_attention_heads,
    ):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            cls._build_simultaneous_text_decoder_arguments(
                    decoder_args,
                    decoder_layers,
                    decoder_embed_dim,
                    decoder_attention_heads,
                ) 
                
            task_decoder = SimultaneousTextDecoder(
                TransformerConfig.from_namespace(decoder_args),
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                )
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                    dictionary=tgt_dict,
                    in_dim=in_dim
                )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    @classmethod
    def _build_unit_ctc_decoder(cls, args, tgt_dict):
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )

        _args = copy.deepcopy(args)
        _args.encoder_embed_dim = args.decoder_embed_dim

        return UnitCTCDecoder(
            _args,
            tgt_dict,
            embed_tokens,
        )
    
    @classmethod
    def _build_streaming_speech_encoder(cls, args):
        encoder = StreamingSpeechEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def _build_text_to_unit_encoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.synthesizer_encoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return TextToUnitEncoder(_args)
    
    @classmethod
    def _build_simultaneous_text_decoder_arguments(cls, args, decoder_layers, decoder_embed_dim=256, decoder_attention_heads=4):
        args.decoder_layers = decoder_layers
        args.decoder_embed_dim = decoder_embed_dim
        args.decoder_attention_heads = decoder_attention_heads
        base_multitask_text_transformer_decoder_arch(args)

    def _build_streaming_mask(self, asr, st, y, src_wait, src_step, tgt_step):
        tgt_len = y.size(1)
        bsz, src_len = st.size()
        idx = torch.arange(0, tgt_len, device=st.device).unsqueeze(0).unsqueeze(2)
        idx = (idx // tgt_step + 1) * src_step + src_wait
        idx = idx.clamp(1, src_len)
        tmp = st.cumsum(dim=-1).unsqueeze(1)
        mask = tmp >= idx
        tmp2 = mask.int() * asr.round().unsqueeze(1)
        tmp2[:, :, -1] = 1
        idx2 = tmp2.max(dim=-1, keepdim=True)[1].clamp(1, src_len)
        if self.encoder.chunk:
            chunk_size = self.encoder.chunk_size
            idx2 = (idx2 // chunk_size + 1) * chunk_size
            idx2 = idx2.clamp(1, src_len)
        tmp3 = torch.arange(0, src_len, device=st.device).unsqueeze(0).unsqueeze(1)

        return tmp3 >= idx2

@register_model_architecture(model_name="streamspeech", arch_name="streamspeech")
def ctc_unity_conformer_architecture_base(args):
    args.conv_version = getattr(args, "conv_version", "convtransformer")
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    s2ut_architecture_base(args)
