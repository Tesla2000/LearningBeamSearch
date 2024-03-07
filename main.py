import random

import numpy as np
import torch

from Config import Config
from model_training.eval_rl import eval_rl
from model_training.train_genetic import train_genetic

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if Config.train:
        for model_type in (
            *Config.model_types,
            *Config.universal_model_types,
            *Config.recurrent_model_types,
        ):
            recurrent = False
            if model_type in Config.model_types:
                models = dict(
                    (tasks, model_type(tasks, Config.m_machines))
                    for tasks in range(Config.min_size, Config.n_tasks + 1)
                )
            else:
                model = model_type()
                models = dict(
                    (tasks, model)
                    for tasks in range(Config.min_size, Config.n_tasks + 1)
                )
                if model_type not in Config.universal_model_types:
                    recurrent = True
            for model in models.values():
                model.to(Config.device)
            train_genetic(
                Config.n_tasks,
                Config.m_machines,
                Config.min_size,
                recurrent,
                models,
                Config.MODEL_RESULTS.joinpath(
                    f"{model_type.__name__}_{Config.n_tasks}_{Config.m_machines}_{Config.min_size}"
                ).open("w"),
            )
            Config.max_tasks = Config.n_tasks + 1
    else:
        eval_rl(
            Config.n_tasks,
            Config.m_machines,
            Config.eval_iterations,
            (
                *Config.model_types,
                *Config.universal_model_types,
            ),
        )
