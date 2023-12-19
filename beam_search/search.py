from itertools import filterfalse, permutations, product
from typing import Callable

import numpy as np

from .Node import Node


class Tree:
    def __init__(self, working_time_matrix: np.array) -> None:
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = Node(None, tuple(), self.m_machines, self.working_time_matrix)

    def branch_and_bound(self):
        def branch_and_bound_cut(
            tree: "Tree", node_value: float, node: Node, n_tasks: int, ub: float
        ) -> bool:
            """Cut is performed only when there is no chance of cutting optimum"""
            not_used_machines = list(
                filterfalse(node.tasks.__contains__, range(n_tasks))
            )
            minimal_value = node_value + np.sum(
                tree.working_time_matrix[not_used_machines, -1]
            )
            return minimal_value > ub

        return self.beam_search(branch_and_bound_cut)

    def beam_search(self, cut: Callable[["Tree", float, Node, int, float], bool]):
        best_node, _ = self.get_best(self.root, self.n_tasks, cut)
        return best_node

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

    def get_best(
        self,
        node: Node,
        n_tasks: int,
        cut: Callable[["Tree", float, Node, int, int | float], bool],
        ub=float("inf"),
    ):
        node_value = node.value
        if cut(self, node_value, node, n_tasks, ub):
            return None, ub
        if len(node.tasks) == n_tasks:
            return node, node_value
        best_node = None
        for task in filterfalse(node.tasks.__contains__, range(n_tasks)):
            new_node, new_ub = self.get_best(
                Node(
                    node, (*node.tasks, task), self.m_machines, self.working_time_matrix
                ),
                n_tasks,
                cut,
                ub,
            )
            if new_ub < ub:
                ub = new_ub
                best_node = new_node
        return best_node, ub
