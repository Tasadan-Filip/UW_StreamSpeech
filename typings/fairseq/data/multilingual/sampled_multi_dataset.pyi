"""
This type stub file was generated by pyright.
"""

from enum import Enum
from fairseq.data import FairseqDataset

def get_time_gap(s, e): # -> str:
    ...

logger = ...
def default_virtual_size_func(datasets, ratios, max_scale_up=...): # -> int:
    ...

class CollateFormat(Enum):
    single = ...
    ordered_dict = ...


class SampledMultiDataset(FairseqDataset):
    """Samples from multiple sub-datasets according to given sampling ratios.
    Args:
        datasets (
            List[~torch.utils.data.Dataset]
            or OrderedDict[str, ~torch.utils.data.Dataset]
        ): datasets
        sampling_ratios (List[float]): list of probability of each dataset to be sampled
            (default: None, which corresponds to concatenating all dataset together).
        seed (int): RNG seed to use (default: 2).
        epoch (int): starting epoch number (default: 1).
        eval_key (str, optional): a key used at evaluation time that causes
            this instance to pass-through batches from *datasets[eval_key]*.
        collate_format (CollateFormat):  collater output format, either CollateFormat.ordered_dict or
            CollateFormat.single (default: CollateFormat.single) where CollateFormat.single configures
            the collater to output batches of data mixed from all sub-datasets,
            and CollateFormat.ordered_dict configures the collater to output a dictionary of batches indexed by keys
            of sub-datasets.
            Note that not all sub-datasets will present in a single batch in both formats.
        virtual_size (int, or callable): the expected virtual size of the dataset (default: default_virtual_size_func).
        split (str): the split of the data, e.g. 'train', 'valid' or 'test'.
        shared_collater (bool): whether or not to all sub-datasets have the same collater.
        shuffle (bool): whether or not to shuffle data (default: True).
    """
    def __init__(self, datasets, sampling_ratios=..., seed=..., epoch=..., eval_key=..., collate_format=..., virtual_size=..., split=..., shared_collater=..., shuffle=...) -> None:
        ...
    
    def setup_sampling(self, sample_ratios, virtual_size): # -> None:
        ...
    
    def adjust_sampling(self, epoch, sampling_ratios, virtual_size): # -> None:
        ...
    
    def random_choice_in_dataset(self, rng, dataset, choice_size):
        ...
    
    def get_virtual_indices(self, rng, datasets, sample_ratios, virtual_size): # -> tuple[NDArray[Any], NDArray[signedinteger[_64Bit]], NDArray[signedinteger[_64Bit]]]:
        ...
    
    def __getitem__(self, index): # -> tuple[int, Any]:
        ...
    
    def num_tokens(self, index): # -> Any:
        ...
    
    def num_tokens_vec(self, indices): # -> Any:
        ...
    
    def size(self, index): # -> ndarray[Any, dtype[Any]]:
        ...
    
    def __len__(self): # -> int | object:
        ...
    
    def collater(self, samples, **extra_args): # -> OrderedDict[Any | int, Any] | dict[str, Tensor | int | dict[str, Tensor | Any] | None] | None:
        """Merge a list of samples to form a mini-batch."""
        ...
    
    @property
    def sizes(self): # -> NDArray[Any]:
        ...
    
    def ordered_indices(self):
        ...
    
    def prefetch(self, indices): # -> None:
        ...
    
    @property
    def can_reuse_epoch_itr_across_epochs(self): # -> Literal[False]:
        ...
    
    def set_epoch(self, epoch): # -> None:
        ...
    
    def filter_indices_by_size(self, indices, max_sizes): # -> tuple[Any, list[Any]] | tuple[Any, Any]:
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        ...
    


