import torch
from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class RecurrentModel(BaseRegressor):
    def __init__(self, n_tasks: int = None, m_machines: int = None):
        from Config import Config
        super().__init__()
        self.n_tasks = n_tasks
        self.m_machines = m_machines
        if self.n_tasks is None:
            self.n_tasks = Config.n_tasks
        if self.m_machines is None:
            self.m_machines = Config.m_machines
        self.hidden_size = self.n_tasks * self.m_machines
        self.relu = nn.LeakyReLU()
        self.gru = nn.GRU(
            input_size=self.m_machines,
            hidden_size=self.hidden_size,
        )
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=1,
        )

    def predict(self, x: torch.Tensor, hn: torch.Tensor):
        out, hn = self.gru(x, hn)
        return self.relu(self.fc(out)), hn

    def update_hn(self, x: torch.Tensor, hn: torch.Tensor):
        out, hn = self.gru(x, hn)
        return hn
