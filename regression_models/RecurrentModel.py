from torch import nn, Tensor
from torch.nn.modules.module import T

from Config import Config
from regression_models.EncodingNetwork import EncodingNetwork
from regression_models.abstract.BaseRegressor import BaseRegressor


class RecurrentModel(BaseRegressor):
    def __init__(self, encoder: EncodingNetwork):
        super().__init__()
        self.encoder = encoder
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=self.encoder.m_machines,
            hidden_size=self.encoder.fc_out_features * self.encoder.out_channels,
        )
        self.fc = nn.Linear(
            in_features=self.encoder.fc_out_features * self.encoder.out_channels,
            out_features=1,
        )
        self.states_by_size = list({} for _ in range(Config.n_tasks + 1))
        self._working_times = {}
        self._working_indexes = {}

    def fill_state(self, working_time_matrix):
        self.states_by_size = list({} for _ in range(Config.n_tasks + 1))
        self.states_by_size[0][tuple()] = (
            self.encoder(Tensor(working_time_matrix).unsqueeze(0))
            .flatten()
            .unsqueeze(0)
        )
        self._working_times = {}
        self._working_indexes = {}
        for index, row in enumerate(working_time_matrix):
            self._working_indexes[tuple(row)] = index
            self._working_times[index] = Tensor(row).unsqueeze(0)

    def forward(self, x):
        preset_indexes = set(
            map(self._working_indexes.get, map(tuple, x.cpu().numpy()[0, 1:]))
        )
        absent_indexes = set(range(len(self._working_indexes))) - preset_indexes
        if tuple(sorted(absent_indexes)) in self.states_by_size[len(absent_indexes)]:
            x = self.fc(
                self.states_by_size[len(absent_indexes)][tuple(sorted(absent_indexes))]
            )
            x = self.relu(x)
            return x
        for index in absent_indexes:
            if (
                indexes := tuple(sorted(absent_indexes.copy() - {index}))
            ) in self.states_by_size[len(absent_indexes.copy() - {index})]:
                prev_state = self.states_by_size[len(indexes)][indexes]
                row = self._working_times[index]
                absent_indexes = tuple(sorted(absent_indexes))
                _, self.states_by_size[len(absent_indexes)][absent_indexes] = self.gru(
                    row.to(Config.device), prev_state
                )
                x = self.fc(self.states_by_size[len(absent_indexes)][absent_indexes])
                x = self.relu(x)
                return x
        for n_elements in range(len(absent_indexes) - 1, -1, -1):
            combination = tuple()
            prev_state = self.states_by_size[0][combination]
            for index in absent_indexes:
                row = self._working_times[index]
                combination = tuple(sorted(set(combination).union({index})))
                _, self.states_by_size[len(combination)][combination] = self.gru(
                    row.to(Config.device), prev_state
                )
                prev_state = self.states_by_size[len(combination)][combination]
            x = self.fc(prev_state)
            x = self.relu(x)
            return x
        raise ValueError

    def eval(self: T) -> T:
        super().eval()
        self.encoder.eval()
        return self

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.encoder.train()
        return self
