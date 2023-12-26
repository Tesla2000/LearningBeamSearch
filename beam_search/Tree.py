from itertools import filterfalse, permutations, product
from typing import Optional

import numpy as np
from torch import nn, Tensor

from .Node import Node


class Tree:
    def __init__(self, working_time_matrix: np.array, models: dict[int, nn.Module] = None) -> None:
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = Node(None, tuple(), self.m_machines, self.working_time_matrix)
        self.models = models or {}

    def branch_and_bound(self) -> Node:
        return self._branch_and_bound(self.root)[0]

    def _branch_and_bound(
        self,
        node: Node,
        ub=float("inf"),
    ) -> tuple[Optional[Node], float]:
        node_value = node.value
        not_used_machines = list(
            filterfalse(node.tasks.__contains__, range(self.n_tasks))
        )
        minimal_value = node_value + np.sum(
            self.working_time_matrix[not_used_machines, -1]
        )
        if minimal_value > ub:
            return None, ub
        if len(node.tasks) == self.n_tasks:
            return node, node_value
        best_node = None
        for task in filterfalse(node.tasks.__contains__, range(self.n_tasks)):
            new_node, new_ub = self._branch_and_bound(
                Node(
                    node, (*node.tasks, task), self.m_machines, self.working_time_matrix
                ),
                ub,
            )
            if new_ub < ub:
                ub = new_ub
                best_node = new_node
        return best_node, ub

    def eval_with_model(self) -> Node:
        return self._eval_with_model(self.root)

    def _eval_with_model(self, node: Node) -> Node:
        if len(node.tasks) == self.n_tasks:
            return node
        model = self.models.get(self.n_tasks - len(node.tasks))
        for task in filterfalse(node.tasks.__contains__, range(self.n_tasks)):
            child_node = Node(
                node, (*node.tasks, task), self.m_machines, self.working_time_matrix
            )
            node.children.append(child_node)
            model_state = Tensor(np.append(child_node.get_state()[-1].reshape(1, -1), self.working_time_matrix[list(filterfalse(child_node.tasks.__contains__, range(self.n_tasks)))], axis=0)).unsqueeze(0)
            child_node.predicted_value = model(model_state) if model else model_state[0, 0, -1]
        return min((self._eval_with_model(child_node) for child_node in sorted(node.children, key=lambda node: node.predicted_value)[:2]), key=lambda item: item.value)

    def brute_force(self):
        task_groups = permutations(range(self.n_tasks))
        best = float('inf')
        for group in task_groups:
            working_time_matrix = self.working_time_matrix[list(group)]
            for row, column in product(range(1, self.n_tasks), range(self.m_machines)):
                if column == 0:
                    working_time_matrix[row, column] += working_time_matrix[row - 1, column]
                else:
                    working_time_matrix[row, column] += max(working_time_matrix[row - 1, column],
                                                            working_time_matrix[row, column - 1])
            best = min(best, working_time_matrix[-1, -1])
        return best
