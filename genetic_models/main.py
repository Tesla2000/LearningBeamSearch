import operator
from functools import partial
from itertools import combinations, starmap
from math import comb
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from numpy.linalg import lstsq
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset


def conv_to_two_d(array: np.array, mask: Sequence[bool]):
    coefs = array[:-1]
    array = np.append(array, coefs ** 2)
    coef_combinations = np.array(tuple(starmap(operator.mul, combinations(coefs, 2))))
    coef_combinations = coef_combinations[mask == 1]
    array = np.append(array, coef_combinations)
    return array


def init_state(n_tasks: int, n_machines: int, calc_best: bool = False) -> tuple[np.array, np.array, Optional[float]]:
    data_file = Path(f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt").open()
    data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=n_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=len(data_maker))
    inputs, labels = next(iter(train_loader))
    labels = labels.numpy()
    inputs = inputs.flatten(start_dim=1).numpy()
    mod_input = np.empty((inputs.shape[0], inputs.shape[1] + 1))
    mod_input[:, :inputs.shape[1]] = inputs
    del inputs
    best_results = None
    mod_input[:, -1] = np.ones_like(mod_input[:, -1])
    if calc_best:
        best_results = calc_result(mod_input, labels)
    return mod_input, labels, best_results

def calc_result(input, target):
    w = lstsq(input, target)[0]
    output = input.dot(w)
    return np.mean(np.abs(target - output))


def init_population(base_input: np.array, pop_size: int = 10, prob_1: float = .002) -> tuple:
    return tuple(
        np.random.choice([0, 1], size=comb(len(base_input[0]) - 1, 2), p=[1 - prob_1, prob_1]) for _ in range(pop_size))


def main():
    n_machines = 25
    for n_tasks in range(3, 11):
        base_input, targets, _ = init_state(n_tasks, n_machines)
        population = init_population(base_input)
        for specimen in population:
            mod_input = np.array(tuple(map(partial(conv_to_two_d, mask=specimen), base_input)))
            result = calc_result(mod_input, targets)
            pass


if __name__ == '__main__':
    np.random.seed(42)
    main()
