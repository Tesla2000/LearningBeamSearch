from itertools import filterfalse, permutations, product, chain
import numpy as np
import torch
from torch import nn, Tensor

from Config import Config


class Tree:
    def __init__(
        self,
        working_time_matrix: Tensor,
        models: dict[int, nn.Module] = None,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = list()
        self.models = models
        if models is None:
            self.models = {}

    def beam_search(self):
        buffer = [self.root]
        for tasks in range(self.n_tasks - 1, 0, -1):
            temp_buffer = np.array(tuple(
                [*node, task] for node in buffer for task in filterfalse(node.__contains__, range(self.n_tasks))))
            if len(temp_buffer) > Config.beta[tasks]:
                if tasks not in self.models:
                    break
                states = self._get_states(temp_buffer)
                headers = states[:, -1].unsqueeze(1)
                remaining_tasks = list(
                    list(filterfalse(tasks.__contains__, range(self.n_tasks))) for tasks in temp_buffer)
                states = self.working_time_matrix[remaining_tasks]
                states = torch.concat((headers, states), dim=1)
                predictions = self.models[tasks](states).flatten()
                temp_buffer = temp_buffer[torch.argsort(predictions)[:Config.beta[tasks]]]
            buffer = temp_buffer
        final_permutations = np.array(tuple(chain.from_iterable(map(lambda remainder: np.append(state, remainder), permutations(
            filterfalse(state.__contains__, range(self.n_tasks)))) for state in temp_buffer)))
        final_states = self._get_states(final_permutations)
        index = np.argmin(final_states[:, -1, -1])
        return final_permutations[index], self._get_states([final_permutations[index]])[0]

    def fast_brute_force(self):
        perms = list(list(permutation) for permutation in permutations(range(self.n_tasks)))
        states = self._get_states(perms)
        index = np.argmin(states[:, -1, -1])
        return perms[index], self._get_states([perms[index]])[0]

    def _get_states(self, perms: np.array):
        states = torch.zeros((len(perms), len(perms[0]) + 1, self.m_machines + 1))
        states[:, 1:, 1:] = self.working_time_matrix[perms]
        for row, column in product(
            range(1, len(perms[0]) + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += torch.maximum(states[:, row - 1, column], states[:, row, column - 1])
        return states[:, 1:, 1:]
