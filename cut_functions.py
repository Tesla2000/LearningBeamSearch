from itertools import filterfalse

import numpy as np

from Node import Node


def branch_and_bound_cut(self: Node, node_value: int, node: Node, n_tasks: int, ub: int) -> bool:
    """Cut is performed only when there is no chance of cutting optimum"""
    not_used_machines = list(filterfalse(node.tasks.__contains__, range(n_tasks)))
    minimal_value = node_value + np.sum(self.working_time_matrix[not_used_machines, -1])
    return minimal_value > ub
