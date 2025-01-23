"""
This type stub file was generated by pyright.
"""

import contextlib
import torch
from typing import Optional
from fairseq.dataclass.configs import DistributedTrainingConfig
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

has_FSDP = ...
class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    fairseq-specific checkpoint saving/loading logic.

    Args:
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """
    def __init__(self, *args, use_sharded_state: bool = ..., **kwargs) -> None:
        ...
    
    @property
    def unwrapped_module(self) -> torch.nn.Module:
        ...
    
    def state_dict(self, destination=..., prefix=..., keep_vars=...): # -> dict[Any, Any]:
        ...
    
    def load_state_dict(self, state_dict, strict=..., model_cfg=...):
        ...
    


class DummyProcessGroup:
    def __init__(self, rank: int, size: int) -> None:
        ...
    
    def rank(self) -> int:
        ...
    
    def size(self) -> int:
        ...
    


@contextlib.contextmanager
def fsdp_enable_wrap(cfg: DistributedTrainingConfig): # -> Generator[None, Any, None]:
    ...

def fsdp_wrap(module, min_num_params: Optional[int] = ..., **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.

    Args:
        module (nn.Module): module to (maybe) wrap
        min_num_params (int, Optional): minimum number of layer params to wrap
    """
    ...

