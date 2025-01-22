from typing import Callable, TypeVar
import typing
from fairseq.tasks import register_task, FairseqTask, FairseqDataclass
from fairseq.models import register_model

TaskType = TypeVar("TaskType", bound=FairseqTask)
TaskRetType = Callable[[TaskType], type[TaskType]]

def register_task_uw(name, dataclass: FairseqDataclass | None = None) -> TaskRetType:
    return typing.cast(TaskRetType, register_task(name, dataclass))


ModelType = TypeVar("ModelType", bound=FairseqTask)
ModelRetType = Callable[[ModelType], type[ModelType]]

def register_model_uw(name, dataclass: FairseqDataclass | None = None) -> ModelRetType:
    return typing.cast(ModelRetType, register_model(name, dataclass))