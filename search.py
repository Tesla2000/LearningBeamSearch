from dataclasses import dataclass, field
from itertools import filterfalse, islice
from typing import Self, Optional

import numpy as np


class Tree:
    def __init__(self, working_time_matrix: np.array) -> None:
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = Node(None, tuple(), self.m_machines, self.working_time_matrix)

    def branch_and_bound(self):
        def cut(node_value: int, node: Node, n_tasks, ub):
            """Cut is performed only when there is no chance of cutting optimum"""
            not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
            minimal_value = node_value + np.sum(self.working_time_matrix[not_used_machines, -1])
            return minimal_value > ub

        def get_best(node, n_tasks, ub=float('inf')):
            node_value = node.value
            if cut(node_value, node, n_tasks, ub):
                return None, ub
            if len(node.tasks) == n_tasks:
                return node, node_value
            best_node = None
            for task in filterfalse(node.tasks.__contains__, range(n_tasks)):
                new_node, new_ub = get_best(Node(node, (*node.tasks, task), self.m_machines, self.working_time_matrix),
                                            n_tasks, ub)
                if new_ub < ub:
                    ub = new_ub
                    best_node = new_node
            return best_node, ub

        best_node, _ = get_best(self.root, self.n_tasks)
        return best_node


@dataclass
class Node:
    parent: Optional[Self]
    tasks: tuple[int]
    m_machines: int
    working_time_matrix: np.array
    children: list[Self] = field(default_factory=list)
    state: np.array = None
    _value: int = 0

    @property
    def value(self) -> int:
        if self._value:
            return self._value
        state = self.get_state()
        if state.shape == (1, 0):
            return 0
        self._value = state[-1, -1]
        return self.value

    def fill_state(self):
        last_task = self.tasks[-1]
        self.state[-1, 0] = self.state[-2, 0] + self.working_time_matrix[last_task, 0]
        for index, work_time in islice(enumerate(self.working_time_matrix[last_task]), 1, None):
            self.state[-1, index] = max(self.state[-1, index - 1], self.state[-2, index]) + self.working_time_matrix[
                last_task, index]

    def get_state(self):
        if self.state is not None:
            return self.state
        if not len(self.tasks):
            return np.array([[]])
        parent_state = self.parent.get_state()
        if parent_state.shape != (1, 0):
            self.state = np.append(parent_state, np.empty((1, self.m_machines)), axis=0)
            self.fill_state()
        else:
            self.state = self.working_time_matrix[self.tasks[-1]:self.tasks[-1] + 1]
        return self.state
