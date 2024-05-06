import random
from collections import defaultdict
from itertools import chain
from statistics import fmean

import numpy as np
import torch

from Config import Config
from beam_search.RecurrentTree import RecurrentTree
from experiments._calc_beta import _calc_beta
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from models.RecurrentModel import RecurrentModel


def recurrent_model_eval(
    iterations: int,
    models: dict[int, RecurrentModel],
    time_constraints: list[int],
    beta_constraints: list[int],
    **_,
):
    _, model = models.popitem()
    del models
    for beta in chain.from_iterable(
            (beta_constraints, (_calc_beta({"": model}, time_constraint, recurrent=True) for time_constraint in time_constraints))):
        results = []
        beta_dict = defaultdict(lambda: beta)
        torch.manual_seed(Config.evaluation_seed)
        torch.cuda.manual_seed(Config.evaluation_seed)
        np.random.seed(Config.evaluation_seed)
        random.seed(Config.evaluation_seed)
        generator = RandomNumberGenerator(Config.evaluation_seed)
        for i in range(iterations):
            working_time_matrix = generate_taillard(generator)
            tree = RecurrentTree(working_time_matrix)
            _, state = tree.beam_search(model, beta_dict)
            results.append(state[-1, -1])
            print(i, type(model).__name__, fmean(results))
        Config.OUTPUT_RL_RESULTS.joinpath(f"{type(model).__name__}_{beta}_{Config.n_tasks}").write_text(str(results))
