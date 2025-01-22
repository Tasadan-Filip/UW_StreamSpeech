"""
This type stub file was generated by pyright.
"""

from torch import nn

logger = ...
class DistributedTimeoutWrapper(nn.Module):
    """
    A wrapper that kills the process if no progress is made within a given
    *timeout*. The timer is reset every time :func:`forward` is called.

    Usage::

        module = DistributedTimeoutWrapper(module, timeout=30)
        x = module(input)
        time.sleep(20)  # safe
        x = module(input)
        time.sleep(45)  # job will be killed before this returns

    Args:
        module (nn.Module): module to wrap
        timeout (int): number of seconds before killing the process
            (set to a value <= 0 to disable the timeout)
        signal (Optional): signal to send once timeout is triggered
    """
    def __init__(self, module: nn.Module, timeout: int, signal=...) -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __getattr__(self, name): # -> Tensor | Module | Any:
        """Forward missing attributes to wrapped module."""
        ...
    
    def stop_timeout(self): # -> None:
        ...
    
    def state_dict(self, *args, **kwargs):
        ...
    
    def load_state_dict(self, *args, **kwargs): # -> _IncompatibleKeys:
        ...
    
    def forward(self, *args, **kwargs): # -> Any:
        ...
    


