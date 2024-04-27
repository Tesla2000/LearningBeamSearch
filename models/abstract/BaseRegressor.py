from abc import abstractmethod, ABC

import torch
from torch import nn


class BaseRegressor(nn.Module, ABC):
    learning_rate = 5e-5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, *args, **kwargs):
        x = x.float()
        x = self.predict(x, *args, **kwargs)
        return x

    def predict(self, x: torch.Tensor, *args, **kwargs):
        pass

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()
