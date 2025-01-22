"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from fairseq.dataclass.configs import DistributedTrainingConfig, FairseqConfig

_USE_MEGATRON = ...
_USE_XLA = ...
logger = ...
def is_master(cfg: DistributedTrainingConfig): # -> bool:
    ...

def infer_init_method(cfg: DistributedTrainingConfig, force_distributed=...): # -> None:
    ...

def distributed_init(cfg: FairseqConfig): # -> int | None:
    ...

def distributed_main(i, main, cfg: FairseqConfig, kwargs): # -> None:
    ...

def call_main(cfg: FairseqConfig, main, **kwargs): # -> None:
    ...

def use_xla(): # -> bool:
    ...

def new_groups(grouped_ranks: List[List[int]]): # -> tuple[Literal['tpu'], List[List[int]]] | object | ProcessGroupMPI | ProcessGroupGloo | ProcessGroupNCCL | ProcessGroupUCC | ProcessGroup:
    ...

def get_rank(group): # -> int:
    ...

def get_world_size(group): # -> int:
    ...

def get_global_group(): # -> tuple[Literal['tpu'], List[List[int]]] | object | ProcessGroupMPI | ProcessGroupGloo | ProcessGroupNCCL | ProcessGroupUCC | ProcessGroup | Any | None:
    ...

def get_global_rank(): # -> Literal[0]:
    ...

def get_global_world_size(): # -> Literal[1]:
    ...

def get_data_parallel_group(): # -> tuple[Literal['tpu'], List[List[int]]] | object | ProcessGroupMPI | ProcessGroupGloo | ProcessGroupNCCL | ProcessGroupUCC | ProcessGroup | Any | None:
    """Get the data parallel group the caller rank belongs to."""
    ...

def get_data_parallel_rank(): # -> int:
    """Return my rank for the data parallel group."""
    ...

def get_data_parallel_world_size(): # -> int:
    """Return world size for the data parallel group."""
    ...

def get_model_parallel_group(): # -> None:
    ...

def get_model_parallel_rank(): # -> int:
    """Return my rank for the model parallel group."""
    ...

def get_model_parallel_world_size(): # -> int:
    """Return world size for the model parallel group."""
    ...

def all_reduce(tensor, group, op=...):
    ...

def broadcast(tensor, src, group): # -> None:
    ...

def all_to_all(tensor, group): # -> Tensor:
    """Perform an all-to-all operation on a 1D Tensor."""
    ...

def all_gather(tensor, group, return_tensor=...): # -> list[Any] | Tensor | list[Any | Tensor]:
    """Perform an all-gather operation."""
    ...

def all_gather_list(data, group=..., max_size=...): # -> list[Any]:
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    ...

def all_reduce_dict(data: Mapping[str, Any], device, group) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group: group of the collective
    """
    ...

def broadcast_tensors(tensors: Optional[List[torch.Tensor]], src_rank: int, group: object, dist_device: Optional[torch.device] = ...) -> List[torch.Tensor]:
    """
    Broadcasts a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    ...

def broadcast_object(obj: Any, src_rank: int, group: object, dist_device: Optional[torch.device] = ...) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    ...

@dataclass(frozen=True)
class _TensorPlaceholder:
    index: int
    ...


