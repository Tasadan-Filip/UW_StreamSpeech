"""
This type stub file was generated by pyright.
"""

import typing as tp
import torch
from functools import lru_cache
from fairseq.data.huffman import HuffmanCoder

class HuffmanMMapIndex:
    """
    keep an index of the offsets in the huffman binary file.
    First a header, then the list of sizes (num tokens) for each instance and finally
    the addresses of each instance.
    """
    _HDR_MAGIC = ...
    _VERSION = ...
    @classmethod
    def writer(cls, path: str, data_len: int): # -> _Writer:
        class _Writer:
            ...
        
        
    
    def __init__(self, path) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __iter__(self): # -> Generator[tuple[ndarray[Any, dtype[signedinteger[_64Bit]]], ndarray[Any, dtype[signedinteger[_32Bit]]]], Any, None]:
        ...
    
    @property
    def data_len(self): # -> Any:
        ...
    
    @property
    def sizes(self): # -> NDArray[signedinteger[_32Bit]]:
        ...
    
    @lru_cache(maxsize=8)
    def __getitem__(self, i): # -> tuple[ndarray[Any, dtype[signedinteger[_64Bit]]], ndarray[Any, dtype[signedinteger[_32Bit]]]]:
        ...
    
    def __len__(self): # -> Any:
        ...
    


def vocab_file_path(prefix_path):
    ...

class HuffmanMMapIndexedDataset(torch.utils.data.Dataset):
    """
    an indexed dataset that use mmap and memoryview to access data from disk
    that was compressed with a HuffmanCoder.
    """
    def __init__(self, prefix_path) -> None:
        ...
    
    def __getstate__(self): # -> None:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    @lru_cache(maxsize=8)
    def __getitem__(self, i): # -> Tensor:
        ...
    
    def __iter__(self): # -> Generator[Tensor, Any, None]:
        ...
    
    def get_symbols(self, i): # -> Generator[str | None, Any, None]:
        ...
    
    @property
    def sizes(self): # -> NDArray[signedinteger[_32Bit]]:
        ...
    
    @property
    def supports_prefetch(self): # -> Literal[False]:
        ...
    
    @property
    def coder(self): # -> HuffmanCoder | None:
        ...
    
    @staticmethod
    def exists(prefix_path): # -> bool:
        ...
    


class HuffmanMMapIndexedDatasetBuilder:
    """
    Helper to build a memory mapped datasets with a huffman encoder.
    You can either open/close this manually or use it as a ContextManager.
    Provide your own coder, it will then be stored alongside the dataset.
    The builder will first write the vocab file, then open the binary file so you can stream
    into it, finally the index will be written when the builder is closed (your index should fit in memory).
    """
    def __init__(self, path_prefix: str, coder: HuffmanCoder) -> None:
        ...
    
    def open(self): # -> None:
        ...
    
    def __enter__(self) -> HuffmanMMapIndexedDatasetBuilder:
        ...
    
    def add_item(self, tokens: tp.List[str]) -> None:
        """
        add a list of tokens to the dataset, they will compressed with the
        provided coder before being written to file.
        """
        ...
    
    def append(self, other_dataset_path_prefix: str) -> None:
        """
        append an existing dataset.
        Beware, if it wasn't built with the same coder, you are in trouble.
        """
        ...
    
    def close(self): # -> None:
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        ...
    


