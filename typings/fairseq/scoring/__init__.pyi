"""
This type stub file was generated by pyright.
"""

import importlib
import os
from abc import ABC, abstractmethod
from fairseq import registry
from omegaconf import DictConfig

class BaseScorer(ABC):
    def __init__(self, cfg) -> None:
        ...
    
    def add_string(self, ref, pred): # -> None:
        ...
    
    @abstractmethod
    def score(self) -> float:
        ...
    
    @abstractmethod
    def result_string(self) -> str:
        ...
    


def build_scorer(choice, tgt_dict): # -> Scorer | Any | None:
    ...

