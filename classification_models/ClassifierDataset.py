import os
import random
from typing import IO

import numpy as np
from torch.utils.data import Dataset

from regression_models.RegressionDataset import NoMoreSamplesException


class ClassifierDataset(Dataset):
    def __init__(self, n_tasks: int, n_machines: int, data_file0: IO, data_file1: IO):
        self.n_tasks = n_tasks
        self.n_machines = n_machines
        self.expected_length = n_tasks * n_machines
        self.data_file0 = data_file0
        self.data_file1 = data_file1
        self.data_file0.seek(0, os.SEEK_END)
        file_size0 = self.data_file0.tell()
        self.data_file0.seek(0)
        self.data_file1.seek(0, os.SEEK_END)
        file_size1 = self.data_file1.tell()
        self.data_file1.seek(0)
        self.prob0 = file_size0 / (file_size0 + file_size1)

    def __len__(self):
        return int(1e9)

    def __getitem__(self, _):
        one = int(random.random() > self.prob0)
        if one:
            io = self.data_file1
        else:
            io = self.data_file0
        prev_state = np.array(
            tuple(map(int, io.readline().split()))
        ).reshape(1, -1)
        if prev_state.shape == (1, 0):
            raise NoMoreSamplesException
        working_time_matrix = np.array(
            tuple(map(ord, io.read(self.expected_length)))
        ).reshape((self.n_tasks, self.n_machines))
        bound = int(io.readline().strip())
        minimal_value = prev_state[0, 0]
        prev_state -= minimal_value
        bound -= minimal_value
        new_order_of_tasks = random.sample(range(len(working_time_matrix)), k=len(working_time_matrix))
        working_time_matrix = working_time_matrix[new_order_of_tasks]
        return (np.append(prev_state, working_time_matrix, axis=0), bound), one
