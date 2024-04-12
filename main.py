import random

import numpy as np
import torch

from Config import Config
from model_training.RandomNumberGenerator import RandomNumberGenerator

if __name__ == "__main__":
    evaluation = getattr(
        getattr(__import__(f'experiments.{Config.series_model_experiment}_eval'), Config.series_model_experiment + '_eval'),
        Config.series_model_experiment + '_eval')
    if Config.train:
        for model_type in (
            *Config.series_models,
            *Config.universal_model_types,
            *Config.recurrent_model_types,
        ):
            if model_type in Config.series_models:
                experiment = getattr(getattr(__import__(f'experiments.{Config.series_model_experiment}'), Config.series_model_experiment),
                                     Config.series_model_experiment)
            elif model_type in Config.recurrent_model_types:
                experiment = getattr(getattr(__import__(f'experiments.{Config.recurrent_model_experiment}'), Config.recurrent_model_experiment),
                                     Config.recurrent_model_experiment)
            else:
                raise ValueError
            torch.manual_seed(Config.seed)
            torch.cuda.manual_seed(Config.seed)
            np.random.seed(Config.seed)
            random.seed(Config.seed)
            generator = RandomNumberGenerator(Config.seed)
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
            for model in models.values():
                model.to(Config.device)
            experiment(
                n_tasks=Config.n_tasks,
                m_machines=Config.m_machines,
                min_size=Config.min_size,
                models=models,
                generator=generator,
                output_file=Config.MODEL_RESULTS.joinpath(
                    f"{model_type.__name__}_{Config.n_tasks}_{Config.m_machines}_{Config.min_size}"
                ).open("w"),
            )
            Config.max_tasks = Config.n_tasks + 1
    else:
        evaluation(
            Config.m_machines,
            Config.eval_iterations,
            (
                *Config.series_models,
                *Config.universal_model_types,
            ),
        )
