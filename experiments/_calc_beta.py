import random
from collections import defaultdict
from time import time

import numpy as np

from Config import Config
from beam_search.GeneticTree import GeneticTree
from beam_search.RecurrentTree import RecurrentTree
from beam_search.Tree import Tree
from experiments._get_beta_genetic_models import _get_beta_genetic_models


def _calc_beta(models: dict, time_constraint: int, recurrent: bool = False, genetic: bool = False) -> int:
    init_beta = 100
    completion_times = {}
    while True:
        completion_time = _get_completion_time(init_beta, models, recurrent, genetic)
        completion_times[init_beta] = completion_time
        if completion_time > time_constraint:
            break
        init_beta *= 2
    step = init_beta // 2
    init_beta -= step
    while step > 0:
        step //= 2
        completion_time = _get_completion_time(init_beta, models, recurrent, genetic)
        completion_times[init_beta] = completion_time
        if completion_time > time_constraint:
            init_beta -= step
        else:
            init_beta += step

    return min(completion_times, key=lambda beta: (time_constraint - completion_times[beta]) ** 2)


def _get_completion_time(init_beta: int, models: dict, recurrent: bool = False, genetic: bool = False) -> float:
    beta = defaultdict(lambda: init_beta)
    working_time_matrix = np.random.randint(low=1, high=100, size=(Config.n_tasks, Config.m_machines))
    if recurrent:
        tree = RecurrentTree(working_time_matrix)
    elif genetic:
        beta_dict, filtered_models = _get_beta_genetic_models(models, init_beta)
        tree = GeneticTree(working_time_matrix, filtered_models)
    else:
        tree = Tree(working_time_matrix, models)
    start = time()
    if recurrent:
        model = models[random.choice(tuple(models.keys()))]
        tree.beam_search(model, beta)
    elif genetic:
        tree.beam_search(beta_dict)
    else:
        tree.beam_search(beta)
    return time() - start
