"""
This type stub file was generated by pyright.
"""

from . import BaseWrapperDataset

class NumelDataset(BaseWrapperDataset):
    def __init__(self, dataset, reduce=...) -> None:
        ...
    
    def __getitem__(self, index): # -> _int:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def collater(self, samples): # -> int | Tensor:
        ...
    


