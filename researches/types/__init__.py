from typing import Callable, Literal, TypeVar
import typing
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.models import BaseFairseqModel, register_model
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.utils import get_activation_fn
from torch import Tensor

TaskType = TypeVar("TaskType", bound=FairseqTask)
TaskRetType = Callable[[TaskType], type[TaskType]]

def register_task_uw(name, dataclass: FairseqDataclass | None = None) -> TaskRetType[TaskType]:
    return typing.cast(TaskRetType[TaskType], register_task(name, dataclass))


ModelType = TypeVar("ModelType", bound=BaseFairseqModel)
ModelRetType = Callable[[ModelType], type[ModelType]]

def register_model_uw(name, dataclass: FairseqDataclass | None = None) -> ModelRetType[ModelType]:
    return typing.cast(ModelRetType[ModelType], register_model(name, dataclass))

ActivationFnName = Literal["relu", "relu_squared", "gelu", "gelu_fast", "gelu_accurate", "tanh", "linear", "swish"]

def get_activation_fn_uw(name: ActivationFnName)-> Callable[[Tensor | int | float], Tensor]:
    return get_activation_fn(name)