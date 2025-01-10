# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch.nn as nn
from ctc_unity.modules.streaming_speech_encoder_convolution_layer_base import StreamingSpeechEncoderConvolutionLayerBase


class StreamingSpeechEncoderConvolutionSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
        chunk_size=None,
    ):
        super(StreamingSpeechEncoderConvolutionSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)

        if chunk_size is None:
            self.conv_layers = nn.ModuleList(
                nn.Conv1d(
                    in_channels if i == 0 else mid_channels // 2,
                    mid_channels if i < self.n_layers - 1 else out_channels * 2,
                    k,
                    stride=2,
                    padding=k // 2,
                )
                for i, k in enumerate(kernel_sizes)
            )
        else:
            self.conv_layers = nn.ModuleList(
                StreamingSpeechEncoderConvolutionLayerBase(
                    in_channels if i == 0 else mid_channels // 2,
                    mid_channels if i < self.n_layers - 1 else out_channels * 2,
                    k,
                    stride=2,
                    # padding=k // 2,
                    chunk_size=chunk_size,
                )
                for i, k in enumerate(kernel_sizes)
            )

    def forward(self, src_tokens, src_lengths):
        _, _, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self._get_out_seq_lens_tensor(src_lengths)

    def _get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out