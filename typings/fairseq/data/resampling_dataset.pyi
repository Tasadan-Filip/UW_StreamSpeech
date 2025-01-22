"""
This type stub file was generated by pyright.
"""

from fairseq.data import BaseWrapperDataset

logger = ...
class ResamplingDataset(BaseWrapperDataset):
    """Randomly samples from a given dataset at each epoch.

    Sampling is done with or without replacement, depending on the "replace"
    parameter.

    Optionally, the epoch size can be rescaled. This is potentially desirable
    to increase per-epoch coverage of the base dataset (since sampling with
    replacement means that many items in the dataset will be left out). In the
    case of sampling without replacement, size_ratio should be strictly less
    than 1.

    Args:
        dataset (~torch.utils.data.Dataset): dataset on which to sample.
        weights (List[float]): list of probability weights
            (default: None, which corresponds to uniform sampling).
        replace (bool): sampling mode; True for "with replacement", or False
            for "without replacement" (default: True)
        size_ratio (float): the ratio to subsample to; must be positive
            (default: 1.0).
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 0).
        epoch (int): starting epoch number (default: 1).
    """
    def __init__(self, dataset, weights=..., replace=..., size_ratio=..., batch_by_size=..., seed=..., epoch=...) -> None:
        ...
    
    def __getitem__(self, index):
        ...
    
    def __len__(self): # -> Any:
        ...
    
    @property
    def sizes(self): # -> list[Any]:
        ...
    
    def num_tokens(self, index):
        ...
    
    def size(self, index):
        ...
    
    def ordered_indices(self): # -> Any | NDArray[signedinteger[Any]]:
        ...
    
    def prefetch(self, indices): # -> None:
        ...
    
    @property
    def can_reuse_epoch_itr_across_epochs(self): # -> Literal[False]:
        ...
    
    def set_epoch(self, epoch): # -> None:
        ...
    


