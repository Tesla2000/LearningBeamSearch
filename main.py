import numpy as np

from generator import RandomNumberGenerator
from search import Tree

if __name__ == '__main__':
    n_tasks = 5
    m_machines = 7
    generator = RandomNumberGenerator()
    working_time_matrix = np.array([[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)])
    tree = Tree(working_time_matrix)
    branch_value = tree.branch_and_bound()
    print(branch_value.value)
