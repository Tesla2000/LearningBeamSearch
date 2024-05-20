import random
from collections import defaultdict
from statistics import fmean

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from experiments._calc_beta import _calc_beta
from experiments.save_results import save_results
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard


def _get_results(beta: int, iterations: int, models: dict, model_type):
    results = []
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
    return results


def series_of_models_eval(
        iterations: int,
        models: dict[int, nn.Module],
        time_constraints: list[int],
        beta_constraints: list[int],
):
    model_type = type(next(iter(models.values())))
    for beta in beta_constraints:
        results = _get_results(beta, iterations, models, model_type)
        save_results(model_type, beta, True, results)
    for time_constraint in time_constraints:
        beta = _calc_beta(models, time_constraint)
        results = _get_results(beta, iterations, models, model_type)
        save_results(model_type, time_constraint, False, results)
