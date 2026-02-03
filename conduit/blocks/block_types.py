from dataclasses import dataclass, is_dataclass
from abc import ABC
from pathlib import Path
from typing import (
    TypeVar,
    Generic,
    Type,
)
from dataclasses import dataclass, is_dataclass
from typing import Type, TypeVar


I = TypeVar("I")
O = TypeVar("O")


@dataclass
class NoOp:
    """Placeholder dataclass for Conduit blocks that take no input or output"""

    pass


class Block(Generic[I, O], ABC):
    def __init__(self, input: Type[I], output: Type[O]):
        self.input = input
        self.output = output

    def __call__(self, data: I) -> O:
        if not is_dataclass(data) or not isinstance(data, self.input):
            raise TypeError(
                f"Expected dataclass instance of {self.input.__name__}, "
                f"got {type(data).__name__}"
            )
        out = self.forward(data)
        if not is_dataclass(out) or not isinstance(out, self.output):
            raise TypeError(
                f"forward() must return {self.output.__name__} dataclass instance, "
                f"got {type(out).__name__}"
            )
        return out
