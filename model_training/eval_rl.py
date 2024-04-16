import re
from collections import defaultdict
from statistics import fmean
from typing import Type

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree


def eval_rl(
    n_tasks: int,
    m_machines: int,
    iterations: int,
    model_types: tuple[Type[nn.Module], ...],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_by_type = {}
    for model_type in model_types:
        models_by_type[model_type] = {}
        if model_type in Config.universal_models:
            for model_path in Config.OUTPUT_RL_MODELS.glob(f"{model_type.__name__}*"):
                model = model_type()
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model.to(device)
                models_by_type[model_type] = defaultdict(lambda: model)
        else:
            for model_path in Config.OUTPUT_RL_MODELS.glob(f"{model_type.__name__}*"):
                tasks = int(re.findall(r"_(\d+)", model_path.name)[0])
                model = model_type(tasks, m_machines)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model.to(device)
                models_by_type[model_type][tasks] = model
    results = dict((model_type, []) for model_type in model_types)
    for i in range(iterations):
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        for model_type, models in models_by_type.items():
            tree = Tree(working_time_matrix, models)
            recurrent = model_type not in (
                *Config.universal_models,
                *Config.series_models,
            )
            _, state = tree.beam_search(Config.minimal_beta, recurrent)
            results[model_type].append(state[-1, -1])
        for model_type, result in results.items():
            print(i, model_type.__name__, fmean(result))
    for model_type, result in results.items():
        Config.OUTPUT_RL_RESULTS.joinpath(model_type.__name__).write_text(str(result))

    return results
