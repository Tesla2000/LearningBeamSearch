import numpy as np

from beam_search.Tree import Tree

if __name__ == '__main__':
    Tree.fast_beam_search(np.random.randint(1, 255, (100, 6, 25)))
