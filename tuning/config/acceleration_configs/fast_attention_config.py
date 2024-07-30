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
class MultipackConfig:

    # 
    effective_batch_size: int = 3840

    # aka max_batch_len
    # https://github.com/instructlab/training/blob/d9237f8df779c737982acc9bfd9e965ccd83cb77/src/instructlab/training/config.py#L126
    max_number_tokens: int  = 60000

@parsable_dataclass
@dataclass
class LossConfig:

    # just put here first, 
    token_averaged_loss: bool = True

@parsable_dataclass
@dataclass
class PaddingFree:
    # just put here first, 
    method: str = "huggingface"

@dataclass
class FastAttentionConfig:

    padding_free: PaddingFree = None

    # to use auto_gptq 4bit lora base layers
    loss_config: LossConfig = None

    # to use auto_gptq 4bit lora base layers
    multipack: MultipackConfig = None

    def __post_init__(self):
        # ensure nested dataclasses initialized
        ensure_nested_dataclasses_initialized(self)