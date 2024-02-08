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

    @staticmethod
    def fast_beam_search(working_time_matrices: np.array) -> np.array:
        n_matrices, n_tasks, m_machines = working_time_matrices.shape
        states = np.zeros((n_matrices, n_tasks + 1, m_machines + 1))
        best_values = np.full(n_matrices, 1e9)
        exploration_stages = np.array(sorted(map(lambda children: np.append(np.array(children), np.full(n_tasks, -1))[:n_tasks], chain.from_iterable((permutations(range(n_tasks), r) for r in range(1, n_tasks + 1)))), key=tuple))
        exploration_states = tuple(exploration_stages[n_tasks-1] for _ in range(n_matrices))
        max_stage = 0

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
