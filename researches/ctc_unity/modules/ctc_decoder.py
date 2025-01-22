from torch import nn

from fairseq.models import FairseqEncoder


class MyCTCDecoder(FairseqEncoder):
    def __init__(self, dictionary, in_dim):
        super().__init__(dictionary)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 1024),
             nn.Sigmoid(),
            nn.Linear(1024, len(dictionary))
        )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_out = self.proj(src_tokens)
        return {"encoder_out": encoder_out}
