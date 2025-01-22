"""
This type stub file was generated by pyright.
"""

from threading import Thread
from typing import Iterator, List

logger = ...
_sentinel = ...
class CountingIterator:
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by ``__len``.
            This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """
    def __init__(self, iterable, start=..., total=...) -> None:
        ...
    
    def __len__(self): # -> Any | int:
        ...
    
    def __iter__(self): # -> Self:
        ...
    
    def __next__(self):
        ...
    
    def has_next(self): # -> Any | bool:
        """Whether the iterator has been exhausted."""
        ...
    
    def skip(self, n): # -> Self:
        """Fast-forward the iterator by skipping n elements."""
        ...
    
    def take(self, n): # -> Self:
        """Truncate the iterator to n elements at most."""
        ...
    


class EpochBatchIterating:
    def __len__(self) -> int:
        ...
    
    @property
    def next_epoch_idx(self):
        ...
    
    def next_epoch_itr(self, shuffle=..., fix_batches_to_gpus=..., set_dataset_epoch=...):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        ...
    
    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        ...
    
    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        ...
    
    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        ...
    
    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        ...
    
    @property
    def first_batch(self): # -> Literal['DUMMY']:
        ...
    


class StreamingEpochBatchIterator(EpochBatchIterating):
    """A steaming-style iterator over a :class:`torch.utils.data.IterableDataset`.

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        max_sentences: batch size
        collate_fn (callable): merges a list of samples to form a mini-batch
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
    """
    def __init__(self, dataset, max_sentences=..., collate_fn=..., epoch=..., num_workers=..., buffer_size=..., timeout=...) -> None:
        ...
    
    @property
    def next_epoch_idx(self): # -> int:
        """Return the epoch index after *next_epoch_itr* is called."""
        ...
    
    def next_epoch_itr(self, shuffle=..., fix_batches_to_gpus=..., set_dataset_epoch=...): # -> CountingIterator:
        ...
    
    def end_of_epoch(self) -> bool:
        ...
    
    @property
    def iterations_in_epoch(self) -> int:
        ...
    
    def state_dict(self): # -> dict[str, int | Any]:
        ...
    
    def load_state_dict(self, state_dict): # -> None:
        ...
    


class FrozenBatchSampler:
    def __init__(self, ordered_batches, epoch, fix_batches_to_gpus, shuffle, initial_offset) -> None:
        ...
    
    def make_batches_for_epoch(self, epoch, offset=...): # -> None:
        ...
    
    def __iter__(self) -> Iterator[List[int]]:
        ...
    
    def __len__(self) -> int:
        ...
    


class EpochBatchIterator(EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler or a callable): an iterator over batches of
            indices, or a callable to create such an iterator (~torch.utils.data.Sampler).
            A callable batch_sampler will be called for each epoch to enable per epoch dynamic
            batch iterators defined by this callable batch_sampler.
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
        disable_shuffling (bool, optional): force disable shuffling
            (default: ``False``).
        skip_remainder_batch (bool, optional): if set, discard the last batch in an epoch
            for the sake of training stability, as the last batch is usually smaller than
                local_batch_size * distributed_word_size (default: ``False``).
        grouped_shuffling (bool, optional): enable shuffling batches in groups
            of num_shards. Ensures that each GPU receives similar length sequences when
            batches are sorted by length.
    """
    def __init__(self, dataset, collate_fn, batch_sampler, seed=..., num_shards=..., shard_id=..., num_workers=..., epoch=..., buffer_size=..., timeout=..., disable_shuffling=..., skip_remainder_batch=..., grouped_shuffling=..., reuse_dataloader=...) -> None:
        ...
    
    @property
    def frozen_batches(self): # -> tuple[Any, ...]:
        ...
    
    @property
    def first_batch(self): # -> Literal['DUMMY']:
        ...
    
    def __len__(self): # -> int:
        ...
    
    @property
    def n(self): # -> Any | int:
        ...
    
    @property
    def next_epoch_idx(self): # -> int:
        """Return the epoch index after *next_epoch_itr* is called."""
        ...
    
    def next_epoch_itr(self, shuffle=..., fix_batches_to_gpus=..., set_dataset_epoch=...): # -> CountingIterator | None:
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        ...
    
    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        ...
    
    @property
    def iterations_in_epoch(self): # -> Any | int:
        """The number of consumed batches in the current epoch."""
        ...
    
    def state_dict(self): # -> dict[str, int | Any | bool]:
        """Returns a dictionary containing a whole state of the iterator."""
        ...
    
    def load_state_dict(self, state_dict): # -> None:
        """Copies the state of the iterator from the given *state_dict*."""
        ...
    
    def ordered_batches(self, epoch, fix_batches_to_gpus, shuffle): # -> list[Any]:
        ...
    


class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
        skip_remainder_batch (bool, optional): if set, discard the last grouped batch in
          each training epoch, as the last grouped batch is usually smaller than
                local_batch_size * distributed_word_size * chunk_size (default: ``False``).
    Attributes:
        n (int): number of elements consumed from this iterator
    """
    def __init__(self, iterable, chunk_size, skip_remainder_batch=...) -> None:
        ...
    


class ShardedIterator(CountingIterator):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).

    Attributes:
        n (int): number of elements consumed from this iterator
    """
    def __init__(self, iterable, num_shards, shard_id, fill_value=..., skip_remainder_batch=...) -> None:
        """
        Args:
            skip_remainder_batch: ignored"""
        ...
    


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len, cuda_device) -> None:
        ...
    
    def run(self): # -> None:
        ...
    


class BufferedIterator:
    def __init__(self, size, iterable) -> None:
        ...
    
    def __iter__(self): # -> Self:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def take(self, n): # -> Self:
        ...
    
    def __next__(self):
        ...
    


class GroupedEpochBatchIterator(EpochBatchIterator):
    """Grouped version of EpochBatchIterator
    It takes several samplers from different datasets.
    Each epoch shuffle the dataset wise sampler individually with different
    random seed. The those sub samplers are combined with into
    one big samplers with deterministic permutation to mix batches from
    different datasets. It will act like EpochBatchIterator but make sure
    1) data from one data set each time
    2) for different workers, they use the same order to fetch the data
    so they will use data from the same dataset everytime
    mult_rate is used for update_freq > 1 case where we want to make sure update_freq
    mini-batches come from same source
    """
    def __init__(self, dataset, collate_fn, batch_samplers, seed=..., num_shards=..., shard_id=..., num_workers=..., epoch=..., mult_rate=..., buffer_size=..., skip_remainder_batch=...) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    @property
    def first_batch(self): # -> Literal['DUMMY']:
        ...
    


