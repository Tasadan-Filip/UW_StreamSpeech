"""
This type stub file was generated by pyright.
"""

from fairseq.tasks import LegacyFairseqTask, register_task

def get_time_gap(s, e): # -> str:
    ...

logger = ...
@register_task("translation_multi_simple_epoch")
class TranslationMultiSimpleEpochTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    @staticmethod
    def add_args(parser): # -> None:
        """Add task-specific arguments to the parser."""
        ...
    
    def __init__(self, args, langs, dicts, training) -> None:
        ...
    
    def check_dicts(self, dicts, source_langs, target_langs): # -> None:
        ...
    
    @classmethod
    def setup_task(cls, args, **kwargs): # -> Self:
        ...
    
    def has_sharded_data(self, split): # -> Literal[False]:
        ...
    
    def load_dataset(self, split, epoch=..., combine=..., **kwargs): # -> None:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        ...
    
    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=...): # -> TransformEosLangPairDataset | LanguagePairDataset:
        ...
    
    def build_generator(self, models, args, seq_gen_cls=..., extra_gen_cls_kwargs=...): # -> SequenceScorer | SequenceGenerator | SequenceGeneratorWithAlignment:
        ...
    
    def build_model(self, args, from_checkpoint=...):
        ...
    
    def valid_step(self, sample, model, criterion): # -> tuple[Any, Any, Any]:
        ...
    
    def inference_step(self, generator, models, sample, prefix_tokens=..., constraints=...):
        ...
    
    def reduce_metrics(self, logging_outputs, criterion): # -> None:
        ...
    
    def max_positions(self): # -> tuple[Any, Any]:
        """Return the max sentence length allowed by the task."""
        ...
    
    @property
    def source_dictionary(self):
        ...
    
    @property
    def target_dictionary(self):
        ...
    
    def create_batch_sampler_func(self, max_positions, ignore_invalid_inputs, max_tokens, max_sentences, required_batch_size_multiple=..., seed=...): # -> Callable[..., Any]:
        ...
    
    def get_batch_iterator(self, dataset, max_tokens=..., max_sentences=..., max_positions=..., ignore_invalid_inputs=..., required_batch_size_multiple=..., seed=..., num_shards=..., shard_id=..., num_workers=..., epoch=..., data_buffer_size=..., disable_iterator_cache=..., skip_remainder_batch=..., grouped_shuffling=..., update_epoch_batch_itr=...): # -> EpochBatchIterator:
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
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
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        ...
    


