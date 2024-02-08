import numpy as np

from beam_search.Tree import Tree

if __name__ == '__main__':
    n_tasks = 9
    Tree(np.random.randint(1, 255, (n_tasks, 25))).fast_beam_search()
