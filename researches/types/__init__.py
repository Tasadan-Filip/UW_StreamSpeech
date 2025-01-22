from typing import Callable, Literal, TypeVar
import typing
from fairseq.tasks import register_task, FairseqTask, FairseqDataclass
from fairseq.models import register_model
from fairseq.utils import get_activation_fn
from torch import Tensor

TaskType = TypeVar("TaskType", bound=FairseqTask)
TaskRetType = Callable[[TaskType], type[TaskType]]

def register_task_uw(name, dataclass: FairseqDataclass | None = None) -> TaskRetType:
    return typing.cast(TaskRetType, register_task(name, dataclass))


ModelType = TypeVar("ModelType", bound=FairseqTask)
ModelRetType = Callable[[ModelType], type[ModelType]]

def register_model_uw(name, dataclass: FairseqDataclass | None = None) -> ModelRetType:
    return typing.cast(ModelRetType, register_model(name, dataclass))


def get_activation_fn_uw(name:  Literal["relu", "relu_squared", "gelu", "gelu_fast", "gelu_accurate", "tanh", "linear", "swish"])-> Callable[[Tensor], Tensor]:
    return get_activation_fn(name)