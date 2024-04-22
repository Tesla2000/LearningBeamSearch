import json
import random
from collections import defaultdict
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
    time_constraint: int,
):
    model_type = type(next(iter(models.values())))
    results = defaultdict(list)
    max_beta = _calc_beta(models, time_constraint)
    torch.manual_seed(Config.evaluation_seed)
    torch.cuda.manual_seed(Config.evaluation_seed)
    np.random.seed(Config.evaluation_seed)
    random.seed(Config.evaluation_seed)
    generator = RandomNumberGenerator(Config.evaluation_seed)
    for _ in range(iterations):
        working_time_matrix = generate_taillard(generator)
        tree = Tree(working_time_matrix, models)
        result = tree.beam_search_eval(max_beta)
        for beta, value in result.items():
            results[beta].append(value)
    Config.OUTPUT_RL_RESULTS.joinpath(model_type.__name__ + "_" + str(time_constraint)).with_suffix('.json').write_text(json.dumps(dict((beta, fmean(values)) for beta, values in results.items())))



