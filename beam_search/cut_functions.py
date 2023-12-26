from itertools import filterfalse

import numpy as np

from .Node import Node
from .Tree import Tree


def sum_cut(tree: "Tree", node_value: float, node: Node, n_tasks: int, ub: float, cut_parameter: float) -> bool:
    not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
    approximated_value = node_value + cut_parameter * np.sum(tree.working_time_matrix[not_used_machines]) + np.sum(
        tree.working_time_matrix[not_used_machines, -1]
    )
    return approximated_value > ub


def ml_cut(tree: "Tree", node_value: float, node: Node, n_tasks: int, ub: float, cut_model) -> bool:
    not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
    if not node.tasks:
        return False
    approximated_value = node_value + cut_model.predict(np.append(tree.working_time_matrix[node.tasks[-1]].reshape(1, -1), tree.working_time_matrix[not_used_machines], axis=0)) + np.sum(
        tree.working_time_matrix[not_used_machines, -1]
    )
    return approximated_value > ub
