from itertools import filterfalse

import numpy as np

from Node import Node
from search import Tree


def sum_cut(tree: "Tree", node_value: float, node: Node, n_tasks: int, ub: float, cut_parameter: float) -> bool:
    """Cut is performed only when there is no chance of cutting optimum"""
    not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
    approximated_value = node_value + cut_parameter * np.sum(tree.working_time_matrix[not_used_machines]) + np.sum(
        tree.working_time_matrix[not_used_machines, -1]
    )
    return approximated_value > ub


def ml_cut(tree: "Tree", node_value: float, node: Node, n_tasks: int, ub: float, cut_parameter: float) -> bool:
    """Cut is performed only when there is no chance of cutting optimum"""
    not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
    approximated_value = node_value +  + np.sum(
        tree.working_time_matrix[not_used_machines, -1]
    )
    return approximated_value > ub
