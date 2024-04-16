import random
import re
from statistics import fmean
from typing import Type

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard


def series_of_models_eval(
    m_machines: int,
    iterations: int,
    model_types: tuple[Type[nn.Module], ...],
):
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    random.seed(Config.seed)
    generator = RandomNumberGenerator(Config.seed)
    models_by_type = {}
    for model_type in model_types:
        models_by_type[model_type] = {}
        for model_path in Config.OUTPUT_RL_MODELS.glob(f"{model_type.__name__}*"):
            tasks = int(re.findall(r"_(\d+)", model_path.name)[0])
            model = model_type(tasks, m_machines)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            model.to(Config.device)
            models_by_type[model_type][tasks] = model
    results = dict((model_type, []) for model_type in model_types)
    for i in range(iterations):
        working_time_matrix = generate_taillard(generator)
        for model_type, models in models_by_type.items():
            tree = Tree(working_time_matrix, models)
            _, state = tree.beam_search(Config.minimal_beta)
            results[model_type].append(state[-1, -1])
        for model_type, result in results.items():
            print(i, model_type.__name__, fmean(result))
    for model_type, result in results.items():
        Config.OUTPUT_RL_RESULTS.joinpath(model_type.__name__).write_text(str(result))

    return results
