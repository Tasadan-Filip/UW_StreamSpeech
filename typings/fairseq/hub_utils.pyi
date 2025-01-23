"""
This type stub file was generated by pyright.
"""

import torch
from typing import Dict, List
from torch import nn

logger = ...
def from_pretrained(model_name_or_path, checkpoint_file=..., data_name_or_path=..., archive_map=..., **kwargs): # -> dict[str, dict[str, Any | str | list[Any] | None] | DictConfig | Any | list[Any] | None]:
    ...

class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """
    def __init__(self, cfg, task, models) -> None:
        ...
    
    @property
    def device(self): # -> device | Tensor | Module:
        ...
    
    def translate(self, sentences: List[str], beam: int = ..., verbose: bool = ..., **kwargs) -> List[str]:
        ...
    
    def sample(self, sentences: List[str], beam: int = ..., verbose: bool = ..., **kwargs) -> List[str]:
        ...
    
    def score(self, sentences: List[str], replace_newline_with_eos: bool = ..., **kwargs): # -> Dict[str, Tensor] | list[Dict[str, Tensor]]:
        ...
    
    def generate(self, tokenized_sentences: List[torch.LongTensor], beam: int = ..., verbose: bool = ..., skip_invalid_size_inputs=..., inference_step_args=..., prefix_allowed_tokens_fn=..., **kwargs) -> List[List[Dict[str, torch.Tensor]]]:
        ...
    
    def encode(self, sentence: str) -> torch.LongTensor:
        ...
    
    def decode(self, tokens: torch.LongTensor) -> str:
        ...
    
    def tokenize(self, sentence: str) -> str:
        ...
    
    def detokenize(self, sentence: str) -> str:
        ...
    
    def apply_bpe(self, sentence: str) -> str:
        ...
    
    def remove_bpe(self, sentence: str) -> str:
        ...
    
    def binarize(self, sentence: str) -> torch.LongTensor:
        ...
    
    def string(self, tokens: torch.LongTensor) -> str:
        ...
    


class BPEHubInterface:
    """PyTorch Hub interface for Byte-Pair Encoding (BPE)."""
    def __init__(self, bpe, **kwargs) -> None:
        ...
    
    def encode(self, sentence: str) -> str:
        ...
    
    def decode(self, sentence: str) -> str:
        ...
    


class TokenizerHubInterface:
    """PyTorch Hub interface for tokenization."""
    def __init__(self, tokenizer, **kwargs) -> None:
        ...
    
    def encode(self, sentence: str) -> str:
        ...
    
    def decode(self, sentence: str) -> str:
        ...
    


