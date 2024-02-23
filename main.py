import random

import numpy as np
import torch

from Config import Config
from model_training.eval_rl import eval_rl
from model_training.train_rl import train_rl

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if Config.train:
        for model_type in (*Config.model_types, *Config.universal_model_types, *Config.recurrent_model_types):
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
            for model in models.values():
                model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            train_rl(
                Config.n_tasks,
                Config.m_machines,
                Config.min_size,
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
            (*Config.model_types, *Config.universal_model_types,),
        )
