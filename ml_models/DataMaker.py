from typing import IO

import numpy as np
from torch.utils.data import Dataset


class DataMaker(Dataset):

    def __init__(
        self,
        n_tasks: int,
        n_machines: int,
        length: int,
        data_file: IO
    ):
        self.n_tasks = n_tasks
        self.n_machines = n_machines
        self.expected_length = n_tasks * n_machines
        self.length = length
        self.data_file = data_file

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        working_time_matrix = np.array(tuple(map(ord, (data := self.data_file.read(self.expected_length))))).reshape((self.n_tasks, self.n_machines))
        best_value = int((value := self.data_file.readline()).strip())
        return working_time_matrix, best_value
