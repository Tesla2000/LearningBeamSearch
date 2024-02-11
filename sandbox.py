import time
from itertools import permutations

import numpy as np
from matplotlib import pyplot as plt

from beam_search.Tree import Tree

if __name__ == '__main__':
    working_time_matrix = np.random.randint(1, 255, (20, 25))
    tree = Tree(working_time_matrix)
    perms = np.array(tuple(permutations(range(9))))
    absolute_results = []
    relative_results = []
    for n_perms in range(1, len(perms)):
        start = time.time()
        for _ in range(1000):
            tree._get_states(perms[:n_perms])
        absolute_results.append(time.time() - start)
        relative_results.append(absolute_results[-1] / n_perms)
        print(absolute_results[-1])
        if n_perms % 100 == 0:
            plt.plot(absolute_results)
            plt.show()
