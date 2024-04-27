from copy import deepcopy
from itertools import product

import numpy as np

from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard


def _calc_c(result_copy):
    for column in range(1, len(result_copy[0])):
        result_copy[0][column] += result_copy[0][column - 1]
    for row in range(1, len(result_copy)):
        result_copy[row][0] += result_copy[row - 1][0]
    for row, column in product(range(1, len(result_copy)), range(1, len(result_copy[0]))):
        result_copy[row][column] += max(result_copy[row - 1][column], result_copy[row][column - 1])
    return result_copy[-1][-1]


def greedy(working_matrix: np.ndarray):
    sorted_working = working_matrix[np.argsort(np.sum(working_matrix, axis=1))]
    result = []
    for task in sorted_working:
        best_i = 0
        best_result = float('inf')
        for i in range(len(result) + 1):
            result_copy = deepcopy(result)
            result_copy.insert(i, task)
            c = _calc_c(deepcopy(result_copy))
            if c < best_result:
                best_result = c
                best_i = i
        result.insert(best_i, task)
    c = _calc_c(deepcopy(result))
    return c


if __name__ == '__main__':
    generator = RandomNumberGenerator(2137)
    print(list(greedy(generate_taillard(generator)) for _ in range(50)))
