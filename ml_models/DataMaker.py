from typing import IO

import numpy as np
from torch.utils.data import Dataset


class DataMaker(Dataset):

    def __init__(
        self,
        n_tasks: int,
        n_machines: int,
        data_file: IO
    ):
        self.n_tasks = n_tasks
        self.n_machines = n_machines
        self.expected_length = n_tasks * n_machines
        self.data_file = data_file

    def __len__(self):
        return int(1e9)

    def __getitem__(self, _):
        prev_state = np.array(tuple(map(int, self.data_file.readline().split()))).reshape(1, -1)
        if prev_state.shape == (1, 0):
            raise NoMoreSamplesException
        working_time_matrix = np.array(tuple(map(ord, self.data_file.read(self.expected_length)))).reshape(
            (self.n_tasks, self.n_machines))
        best_value = int(self.data_file.readline().strip())
        minimal_value = prev_state[0, 0]
        prev_state -= minimal_value
        best_value -= minimal_value
        return np.append(prev_state, working_time_matrix, axis=0), best_value


class NoMoreSamplesException(Exception):
    pass
