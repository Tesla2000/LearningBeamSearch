import torch
from torch import nn

from regression_models import MultilayerPerceptron


class WideMultilayerPerceptron(MultilayerPerceptron):
    def __init__(self, n_tasks: int, n_machines: int, hidden_size: int = 256, **_):
        super(WideMultilayerPerceptron, self).__init__(n_tasks, n_machines, hidden_size)
        self.dense2 = nn.Linear(
            in_features=hidden_size + (n_tasks + 1) * n_machines, out_features=1
        )

    def predict(self, x):
        x = self.flatten(x)
        out = self.dense1(x)
        x = torch.concat((x, out), dim=1)
        return self.dense2(x)
