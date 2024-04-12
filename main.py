import random

import numpy as np
import torch

from Config import Config
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.eval_rl import eval_rl

if __name__ == "__main__":
    experiment = getattr(getattr(__import__(f'experiments.{Config.experiment}'), Config.experiment), Config.experiment)
    if Config.train:
        for model_type in (
            *Config.series_models,
            *Config.universal_model_types,
            *Config.recurrent_model_types,
        ):
            torch.manual_seed(Config.seed)
            np.random.seed(Config.seed)
            random.seed(Config.seed)
            generator = RandomNumberGenerator(Config.seed)
            recurrent = False
            if model_type in Config.series_models:
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
            experiment(
                n_tasks=Config.n_tasks,
                m_machines=Config.m_machines,
                min_size=Config.min_size,
                recurrent=recurrent,
                models=models,
                generator=generator,
                output_file=Config.MODEL_RESULTS.joinpath(
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
                *Config.series_models,
                *Config.universal_model_types,
            ),
        )
