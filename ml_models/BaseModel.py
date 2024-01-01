from abc import abstractmethod, ABC

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    def _min_value(self, x):
        return torch.sum(x[:, :, -1], axis=1).reshape(-1, 1)

    def forward(self, x):
        x = x.float()
        min_value = self._min_value(x)
        x = self.predict(x)
        return x + min_value

    @abstractmethod
    def predict(self, x):
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.drop2(x)
        return self.dense2(x)
