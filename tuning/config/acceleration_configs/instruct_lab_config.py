# Standard
from dataclasses import dataclass
from typing import List

# Local
from .utils import (
    ensure_nested_dataclasses_initialized,
    parsable_dataclass,
)

@parsable_dataclass
@dataclass
class PaddingFree:
    # just put here first, 
    method: str = "huggingface"

@dataclass
class InstructLabConfig:

    padding_free: PaddingFree = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)