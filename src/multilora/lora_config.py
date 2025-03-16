from dataclasses import dataclass
from typing import List, Callable

@dataclass
class LoRAConfig:
    pass


@dataclass
class LoRAEmptyConfig(LoRAConfig):
    target_modules: List[str]
    r = 8
    alpha = 1.0
    dropout = 1.0


@dataclass
class LoRAPretrainedConfig(LoRAConfig):
    factory: Callable
    kwargs: dict


@dataclass
class LLMConfig:
    factory: Callable
    kwargs: dict


@dataclass
class LoRASetup:
    adapter_list: List[LoRAConfig]
    llm: LLMConfig