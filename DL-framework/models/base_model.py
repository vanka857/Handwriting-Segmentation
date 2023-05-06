import torch.nn as nn
from abc import ABC, abstractclassmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super(nn.Module, self).__init__(*args, **kwargs)
