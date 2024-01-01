import torch
from torch import nn

from ml_models import ConvModel


class WideConvModel(ConvModel):

    def __init__(self, n_tasks: int, n_machines: int, hidden_size: int = 256):
        super(WideConvModel, self).__init__(n_tasks, n_machines)
        self.dense1 = nn.Linear(in_features=(n_tasks + 1) * n_machines * 9, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size + (n_tasks + 1) * n_machines * 9, out_features=1)

    def predict(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.dense1(x)
        x = torch.concat((x, out), dim=1)
        return self.dense2(x)
