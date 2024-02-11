import operator
from functools import reduce
from itertools import filterfalse, permutations, product, chain
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from Config import Config


class Tree:
    def __init__(
        self,
        working_time_matrix: np.array,
        models: dict[int, nn.Module] = None,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = list()
        self.models = models
        if models is None:
            self.models = {}
        self.perms = None

    def beam_search(self):
        buffer = [self.root]
        for tasks in range(self.n_tasks - 1, 0, -1):
            temp_buffer = np.array(tuple(
                [*node, task] for node in buffer for task in filterfalse(node.__contains__, range(self.n_tasks))))
            if len(temp_buffer) > Config.beta[tasks]:
                if tasks not in self.models:
                    break
                states = self._get_states(temp_buffer)
                headers = states[:, [-1]]
                remaining_tasks = list(
                    list(filterfalse(tasks.__contains__, range(self.n_tasks))) for tasks in temp_buffer)
                states = self.working_time_matrix[remaining_tasks]
                states = np.append(headers, states, axis=1)
                predictions = self.models[tasks](Tensor(states)).flatten()
                temp_buffer = temp_buffer[torch.argsort(predictions)[:Config.beta[tasks]]]
            buffer = temp_buffer
        final_permutations = np.array(
            tuple(chain.from_iterable(map(lambda remainder: np.append(state, remainder), permutations(
                filterfalse(state.__contains__, range(self.n_tasks)))) for state in temp_buffer)))
        final_states = self._get_states(final_permutations)
        index = np.argmin(final_states[:, -1, -1])
        return final_permutations[index], self._get_states([final_permutations[index]])[0]

    def fast_branch_and_bound(self, seed: int = 3, tasks: Optional[np.array] = None, ub=float('inf')):
        if tasks is None:
            perms = np.array(tuple(permutations(range(self.n_tasks), seed)))
        else:
            perms = np.append(np.array(reduce(operator.mul, (max(1, self.n_tasks - len(tasks) - i) for i in range(seed)))*[tasks]), np.array(tuple(permutations(filterfalse(tasks.__contains__, range(self.n_tasks)), min(self.n_tasks - len(tasks), seed)))), axis=1)
        best_order = None
        if len(perms[0]) != self.n_tasks:
            states = self._get_states(perms)
            if ub != float('inf'):
                lbs = self._get_lbs(states, perms)
                states = states[np.where(lbs < ub)]
            for state_index in np.argsort(states[:, -1, -1]):
                tasks = perms[state_index]
                order, value = self.fast_branch_and_bound(seed, tasks, ub)
                if value < ub:
                    ub = value
                    best_order = order
            return best_order, ub
        else:
            states = self._get_states(perms)
            index = np.argmin(states[:, -1, -1])
            return perms[index], states[index, -1, -1]

    def fast_brute_force(self):
        if self.perms is None:
            self.perms = list(list(permutation) for permutation in permutations(range(self.n_tasks)))
        states = self._get_states(self.perms)
        index = np.argmin(states[:, -1, -1])
        return self.perms[index], self._get_states([self.perms[index]])[0]

    def _get_states(self, perms: np.array):
        states = np.zeros((len(perms), len(perms[0]) + 1, self.m_machines + 1))
        states[:, 1:, 1:] = self.working_time_matrix[perms]
        for row, column in product(
            range(1, len(perms[0]) + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += np.maximum(states[:, row - 1, column], states[:, row, column - 1])
        return states[:, 1:, 1:]

    def _get_lbs(self, states: np.array, perms: np.array):
        return states[:, -1, -1] + np.sum(self.working_time_matrix[:, -1]) - np.sum(self.working_time_matrix[perms][:, :, -1], axis=1)
