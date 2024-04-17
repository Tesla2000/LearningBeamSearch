import random
from collections import defaultdict
from statistics import fmean
from time import time

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard


def series_of_models_eval(
    iterations: int,
    models: dict[int, nn.Module],
    time_constraints: list[int],
):
    model_type = type(next(iter(models.values())))
    results = []
    for time_constraint in time_constraints:
        beta = _calc_beta(models, time_constraint)
        beta_dict = defaultdict(lambda: beta)
        torch.manual_seed(Config.evaluation_seed)
        torch.cuda.manual_seed(Config.evaluation_seed)
        np.random.seed(Config.evaluation_seed)
        random.seed(Config.evaluation_seed)
        generator = RandomNumberGenerator(Config.evaluation_seed)
        for i in range(iterations):
            working_time_matrix = generate_taillard(generator)
            tree = Tree(working_time_matrix, models)
            _, state = tree.beam_search(beta_dict)
            results.append(state[-1, -1])
            print(i, model_type.__name__, fmean(results))
    Config.OUTPUT_RL_RESULTS.joinpath(model_type.__name__).write_text(str(results))
    return results


def _calc_beta(models: dict, time_constraint: int) -> int:
    init_beta = 100
    completion_times = {}
    while True:
        completion_time = _get_completion_time(init_beta, models)
        completion_times[init_beta] = completion_time
        if completion_time > time_constraint:
            break
        init_beta *= 2
    step = init_beta // 2
    init_beta -= step
    while step > 0:
        step //= 2
        completion_time = _get_completion_time(init_beta, models)
        completion_times[init_beta] = completion_time
        if completion_time > time_constraint:
            init_beta -= step
        else:
            init_beta += step

    return min(completion_times, key=lambda beta: (time_constraint - completion_times[beta]) ** 2)


def _get_completion_time(init_beta: int, models: dict) -> float:
    beta = defaultdict(lambda: init_beta)
    working_time_matrix = np.random.randint(low=1, high=100, size=(Config.n_tasks, Config.m_machines))
    tree = Tree(working_time_matrix, models)
    start = time()
    tree.beam_search(beta)
    return time() - start
