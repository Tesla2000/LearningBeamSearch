from dataclasses import dataclass, field
from itertools import filterfalse, permutations, product, chain
from math import factorial
from typing import Optional
import torch
import numpy as np

from .Node import Node


class Tree:
    def __init__(
        self,
        working_time_matrix: np.array,
        beta: float = 0.5,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = Node(None, tuple(), self.m_machines, self.working_time_matrix)
        self.beta = beta

    def _cut(
        self, node: Node, node_value: float, not_used_machines: list[int], ub: float
    ) -> bool:
        return node_value + np.sum(self.working_time_matrix[not_used_machines, -1]) > ub

    def _beam_search(
        self,
        node: Node,
        ub=float("inf"),
    ) -> tuple[Optional[Node], float]:
        node_value = node.value
        not_used_machines = list(
            filterfalse(node.tasks.__contains__, range(self.n_tasks))
        )
        if self._cut(node, node_value, not_used_machines, ub):
            return None, ub
        if len(node.tasks) == self.n_tasks:
            return node, node_value
        best_node = None
        for task in filterfalse(node.tasks.__contains__, range(self.n_tasks)):
            child_node = Node(
                node, (*node.tasks, task), self.m_machines, self.working_time_matrix
            )
            new_node, new_ub = self._beam_search(
                child_node,
                ub,
            )
            node.children.append(child_node)
            if new_ub < ub:
                ub = new_ub
                best_node = new_node
        return best_node, ub

    def beam_search(self) -> Node:
        return self._beam_search(self.root)[0]

    def fast_beam_search(self, r: int = 5) -> Node:
        permutes = list(permutations(range(self.n_tasks), r))
        working_time_matrices = np.zeros((len(permutes), self.n_tasks + 1, self.m_machines + 1))
        working_time_matrices[:, 1:r + 1, 1:] = self.working_time_matrix[permutes]
        nodes = list(
            Node(None, tasks, self.m_machines, working_time_matrices[index]) for index, tasks in enumerate(permutes))
        ub = float('inf')
        pass
        # ordered_exploration_stages = np.array(tuple(map(lambda stage: (stage + n_tasks * [-1])[:n_tasks], sorted(
        #     map(list, chain.from_iterable((permutations(range(n_tasks), r) for r in range(1, n_tasks + 1))))))))
        # n_exploration_stages = len(ordered_exploration_stages)
        # exploration_indexes = np.full(n_matrices, n_tasks - 1)
        # while True:
        #     states = np.zeros((n_matrices, n_tasks + 1, m_machines + 1))
        #     exploration_states = ordered_exploration_stages[exploration_indexes]
        #     states[:, 1:, 1:] = np.array(
        #         tuple(matrix[state] for matrix, state in
        #               zip(working_time_matrices, exploration_states)))
        #     lb = np.sum(states[:, 1], axis=1) + np.sum(states[:, 2:, -1], axis=1)
        #     if np.argwhere(lb > ub).shape != (0, 1):
        #         for index in np.argwhere(lb > ub):
        #             index = index[0]
        #             exploration_index = exploration_indexes[index]
        #             empty = np.count_nonzero(ordered_exploration_stages[exploration_index] == -1)
        #             for i in range(exploration_index + 1, n_exploration_stages):
        #                 if empty == np.count_nonzero(ordered_exploration_stages[i] == -1):
        #                     exploration_indexes[index] = i
        #                     break
        #
        #     for row, column in product(
        #         range(1, n_tasks + 1), range(1, m_machines + 1)
        #     ):
        #         states[:, row, column] += np.maximum(states[:, row - 1, column], states[:, row, column - 1])
        #     ub = np.minimum(ub, states[:, -1, -1])
        #     exploration_indexes += 1
        #     if np.where(exploration_indexes == n_exploration_stages):
        #         pass

    def brute_force(self):
        best_value = float("inf")
        for permutation in permutations(range(self.n_tasks)):
            state = self.working_time_matrix[list(permutation)]
            state[0] = np.add.accumulate(state[0])
            state[:, 0] = np.add.accumulate(state[:, 0])
            for row, column in product(
                range(1, self.n_tasks), range(1, self.m_machines)
            ):
                state[row, column] += max(
                    state[row - 1, column], state[row, column - 1]
                )
            value = state[-1, -1]
            if value < best_value:
                best_value = value
        return best_value

    def fast_brute_force(self):
        states = np.zeros((factorial(self.n_tasks), self.n_tasks + 1, self.m_machines + 1))
        states[:, 1:, 1:] = self.working_time_matrix[
            list(list(permutation) for permutation in permutations(range(self.n_tasks)))]
        for row, column in product(
            range(1, self.n_tasks + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += np.maximum(states[:, row - 1, column], states[:, row, column - 1])
        return np.min(states[:, -1, -1])

    def faster_brute_force(self):
        states = torch.zeros((factorial(self.n_tasks), self.n_tasks + 1, self.m_machines + 1))
        states[:, 1:, 1:] = torch.Tensor(self.working_time_matrix)[
            list(list(permutation) for permutation in permutations(range(self.n_tasks)))]
        for row, column in product(
            range(1, self.n_tasks + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += torch.maximum(states[:, row - 1, column], states[:, row, column - 1])
        return torch.min(states[:, -1, -1])
