import torch
from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class SumRegressor(BaseRegressor):
    learning_rate = 1e-3

    def __init__(self, **_):
        super(SumRegressor, self).__init__()
        self.fc = nn.Linear(1, 1)

    def predict(self, x):
        x = torch.sum(x[:, 1:, :-1], dim=[1, 2]).reshape(-1, 1)
        return self.fc(x)
