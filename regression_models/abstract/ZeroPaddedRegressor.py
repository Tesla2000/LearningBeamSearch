from abc import abstractmethod, ABC

import torch
from torch import nn

from Config import Config


class ZeroPaddedRegressor(nn.Module, ABC):
    max_n_task: int = Config.n_tasks
    learning_rate = 5e-5

    def forward(self, x):
        x = x.float()
        zeros = torch.zeros((x.shape[0], self.max_n_task + 1, x.shape[2])).to(
            Config.device
        )
        zeros[:, : x.shape[1], :] = x
        x = zeros
        x = self.predict(x)
        return x

    @abstractmethod
    def predict(self, x):
        pass

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()
