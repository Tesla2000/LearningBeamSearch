from abc import abstractmethod, ABC

from torch import nn


class BaseRegressor(nn.Module, ABC):
    learning_rate = 5e-5

    def forward(self, x):
        x = x.float()
        x = self.predict(x)
        return x

    def predict(self, x):
        pass

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()
