import random
from collections import defaultdict
from itertools import chain
from statistics import fmean

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from experiments._calc_beta import _calc_beta
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard


def series_of_models_eval(
        iterations: int,
        models: dict[int, nn.Module],
        time_constraints: list[int],
        beta_constraints: list[int],
):
    model_type = type(next(iter(models.values())))
    for beta in chain.from_iterable(
            (beta_constraints, (_calc_beta(models, time_constraint) for time_constraint in time_constraints))):
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
        Config.OUTPUT_RL_RESULTS.joinpath(f"{model_type.__name__}_{beta}_{Config.n_tasks}").write_text(str(results))