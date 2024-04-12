import torch
from torch import nn

from regression_models import ConvRegressor


class WideConvRegressor(ConvRegressor):
    def __init__(self, n_tasks: int, m_machines: int, hidden_size: int = 256):
        super(WideConvRegressor, self).__init__(n_tasks, m_machines)
        self.dense1 = nn.Linear(
            in_features=(n_tasks + 1) * m_machines * 9, out_features=hidden_size
        )
        self.dense2 = nn.Linear(
            in_features=hidden_size + (n_tasks + 1) * m_machines * 9, out_features=1
        )

    def predict(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        out = self.dense1(x)
        x = torch.concat((x, out), dim=1)
        return self.dense2(x)
