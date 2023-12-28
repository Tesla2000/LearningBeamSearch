from itertools import filterfalse
from typing import Optional

import numpy as np
from torch import nn, Tensor

from .Node import Node


class Tree:
    def __init__(self, working_time_matrix: np.array, models: dict[int, nn.Module] = None) -> None:
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = Node(None, tuple(), self.m_machines, self.working_time_matrix)
        self.models = models

    def _get_minimal_value(self, node: Node, node_value: float, not_used_machines: list[int]) -> float:
        return node_value + np.sum(
            self.working_time_matrix[not_used_machines, -1]
        ) if (model := self.models.get(len(not_used_machines))) is None else float(model(
            Tensor(np.append((np.zeros((1, node.m_machines)) if node.state is None else node.state)[-1].reshape(1, -1),
                             self.working_time_matrix[not_used_machines], axis=0)).unsqueeze(0)))

    def _eval_with_model(
        self,
        node: Node,
        ub=float("inf"),
    ) -> tuple[Optional[Node], float]:
        node_value = node.value
        not_used_machines = list(
            filterfalse(node.tasks.__contains__, range(self.n_tasks))
        )
        minimal_value = self._get_minimal_value(node, node_value, not_used_machines)
        if minimal_value > ub:
            return None, ub
        if len(node.tasks) == self.n_tasks:
            return node, node_value
        best_node = None
        for task in filterfalse(node.tasks.__contains__, range(self.n_tasks)):
            child_node = Node(
                node, (*node.tasks, task), self.m_machines, self.working_time_matrix
            )
            new_node, new_ub = self._eval_with_model(
                child_node,
                ub,
            )
            node.children.append(child_node)
            if new_ub < ub:
                ub = new_ub
                best_node = new_node
        return best_node, ub

    def eval_with_model(self) -> Node:
        return self._eval_with_model(self.root)[0]
