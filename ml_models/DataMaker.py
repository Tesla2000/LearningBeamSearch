from itertools import accumulate

import numpy as np
from torch.utils.data import Dataset

from beam_search.generator import RandomNumberGenerator
from beam_search.Tree import Tree


class DataMaker(Dataset):

    def __init__(
        self,
        n_tasks: int,
        m_machines: int,
        rows: int,
        length: int,
    ):
        self.rows = rows
        self.n_tasks = n_tasks
        self.m_machines = m_machines
        self.length = length
        self.generator = RandomNumberGenerator()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        working_time_matrix = np.array(
            [[self.generator.nextInt(1, 99) for _ in range(self.m_machines)] for _ in range(self.rows)]
        )
        working_time_matrix[0] = np.array(tuple(accumulate(self.generator.nextInt(1, 40) for _ in range(self.m_machines))))
        working_time_matrix[0] -= np.min(working_time_matrix[0])
        self.tree = Tree(working_time_matrix)
        best_node = self.tree.branch_and_bound()
        # test_value = self.tree.brute_force()
        # assert best_node.value == test_value
        return working_time_matrix, best_node.value - np.max(working_time_matrix[0])
