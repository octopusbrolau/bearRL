import torch
from typing import Optional, Union
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self.extractor: torch.nn.Module = None
        self.actor: torch.nn.Module = None
        self.critic: torch.nn.Module = None

    @abstractmethod
    def set_device(self, device: Union[str, torch.device] = "cpu"):
        pass

    def save(self):
        pass

    def load(self):
        pass
