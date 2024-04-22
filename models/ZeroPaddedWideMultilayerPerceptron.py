import torch
from torch import nn

from models.ZeroPaddedMultilayerPerceptron import ZeroPaddedMultilayerPerceptron


class ZeroPaddedWideMultilayerPerceptron(ZeroPaddedMultilayerPerceptron):
    def __init__(self, hidden_size: int = 64, **_):
        from Config import Config
        super().__init__(hidden_size)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.dense2 = nn.Linear(in_features=hidden_size + (self.max_n_task + 1) * Config.m_machines, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        out = self.dense1(x)
        x = torch.concat((x, out), dim=1)
        x = self.relu(x)
        x = self.dense2(x)
        return self.relu(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)
