import random
from itertools import chain
from statistics import fmean

import numpy as np
import torch

from Config import Config
from beam_search.GeneticTree import GeneticTree
from experiments._calc_beta import _calc_beta
from experiments._get_beta_genetic_models import _get_beta_genetic_models
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from models.GeneticRegressor import GeneticRegressor


def genetic_series_of_models_eval(
        iterations: int,
        models_lists: dict[int, dict[GeneticRegressor, float]],
        time_constraints: list[int],
        beta_constraints: list[int],
        **_,
):
    for beta in chain.from_iterable(
            (beta_constraints,
             (_calc_beta(models_lists, time_constraint, genetic=True) for time_constraint in time_constraints))):
        beta_dict, filtered_models = _get_beta_genetic_models(models_lists, beta)
        results = []
        torch.manual_seed(Config.evaluation_seed)
        torch.cuda.manual_seed(Config.evaluation_seed)
        np.random.seed(Config.evaluation_seed)
        random.seed(Config.evaluation_seed)
        generator = RandomNumberGenerator(Config.evaluation_seed)
        for i in range(iterations):
            working_time_matrix = generate_taillard(generator)
            tree = GeneticTree(working_time_matrix, filtered_models)
            _, state = tree.beam_search(beta_dict)[0]
            results.append(state[-1, -1])
            print(i, GeneticRegressor.__name__, fmean(results))
        Config.OUTPUT_RL_RESULTS.joinpath(f"{GeneticRegressor.__name__}_{beta}_{Config.n_tasks}").write_text(str(results))


