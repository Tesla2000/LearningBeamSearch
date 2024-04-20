import torch
from torch import nn

from models.abstract.BaseRegressor import BaseRegressor


class GRURegressor(BaseRegressor):
    def __init__(
        self, m_machines: int, num_layers: int = 2, hidden_size: int = 256, **_
    ):
        super(GRURegressor, self).__init__()
        self.m_machines = m_machines
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(m_machines, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def predict(self, x):
        h0 = torch.zeros(self.num_layers, x.n_tasks(0), self.hidden_size)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)
