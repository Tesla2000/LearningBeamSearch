from itertools import combinations_with_replacement
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from numpy.linalg import lstsq
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset


def conv_to_two_d(array: np.array, mask: Sequence[bool]):
    coef_combinations = np.array(tuple(combinations_with_replacement(range(len(array[0]) - 1), 2)))
    coef_combinations = coef_combinations[mask == 1]
    return np.concatenate((array, *tuple(
        np.multiply(array[:, combo[0]], array[:, combo[1]]).reshape(-1, 1) for combo in coef_combinations)), axis=1)


def init_state(
    n_tasks: int, n_machines: int, batch_size: int = None, calc_best: bool = False
) -> tuple[np.array, np.array, Optional[float]]:
    data_file = Path(
        f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt"
    ).open()
    data_maker = RegressionDataset(
        n_tasks=n_tasks, n_machines=n_machines, data_file=data_file
    )
    if batch_size is None:
        batch_size = len(data_maker)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    inputs, labels = next(iter(train_loader))
    labels = labels.numpy()
    inputs = inputs.flatten(start_dim=1).numpy()
    mod_input = np.empty((inputs.shape[0], inputs.shape[1] + 1))
    mod_input[:, : inputs.shape[1]] = inputs
    del inputs
    best_results = None
    mod_input[:, -1] = np.ones_like(mod_input[:, -1])
    if calc_best:
        best_results = calc_result(mod_input, labels)
    return mod_input, labels, best_results


def calc_result(input, target):
    w = lstsq(input, target, rcond=-1)[0]
    output = input.dot(w)
    return np.mean(np.abs(target - output))
