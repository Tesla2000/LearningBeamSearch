from abc import abstractmethod, ABC

import torch
from torch import nn


class BaseRegressor(nn.Module, ABC):
    learning_rate = 1e-4

    def _min_value(self, x):
        return torch.sum(x[:, :, -1], axis=1).reshape(-1, 1)

    def forward(self, x):
        x = x.float()
        min_value = self._min_value(x)
        x = self.predict(x)
        return x + min_value

    @abstractmethod
    def predict(self, x):
        pass

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()