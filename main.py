import time
from functools import partial

import numpy as np

from beam_search.cut_functions import sum_cut, ml_cut
from beam_search.generator import RandomNumberGenerator
from beam_search.search import Tree

if __name__ == "__main__":
    n_tasks = 7
    m_machines = 10
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]
    )
    tree = Tree(working_time_matrix)
    start = time.time()
    beam_value = tree.beam_search(partial(sum_cut, cut_parameter=.05))
    # beam_value = tree.beam_search(partial(ml_cut, cut_model=None))
    print(beam_value.value)
    print(time.time() - start)
    start = time.time()
    branch_value = tree.branch_and_bound()
    print(branch_value.value)
    print(time.time() - start)
