import random
import re

import numpy as np
import torch

from Config import Config
from model_training.RandomNumberGenerator import RandomNumberGenerator

if __name__ == "__main__":
    if Config.train:
        for model_type in (
            *Config.series_models,
            *Config.universal_models,
            *Config.recurrent_models,
            *Config.genetic_models,
        ):
            if model_type in Config.recurrent_models:
                experiment = getattr(getattr(__import__(f'experiments.{Config.recurrent_model_experiment}'), Config.recurrent_model_experiment),
                                     Config.recurrent_model_experiment)
            elif model_type in Config.genetic_models:
                experiment = getattr(getattr(__import__(f'experiments.{Config.genetic_model_experiment}'), Config.genetic_model_experiment),
                                     Config.genetic_model_experiment)
            else:
                experiment = getattr(getattr(__import__(f'experiments.{Config.series_model_experiment}'), Config.series_model_experiment),
                                     Config.series_model_experiment)
            torch.manual_seed(Config.seed)
            torch.cuda.manual_seed(Config.seed)
            np.random.seed(Config.seed)
            random.seed(Config.seed)
            generator = RandomNumberGenerator(Config.seed)
            models = {}
            if model_type in Config.series_models:
                models = dict(
                    (tasks, model_type(tasks, Config.m_machines))
                    for tasks in range(Config.min_size, Config.n_tasks + 1)
                )
            elif model_type in (*Config.recurrent_models, *Config.universal_models):
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
                output_file=Config.MODEL_TRAIN_LOG.joinpath(
                    f"{model_type.__name__}_{Config.n_tasks}_{Config.m_machines}_{Config.min_size}"
                ).open("w"),
            )
            Config.max_tasks = Config.n_tasks + 1
    else:
        for model_type in (
            *Config.series_models,
            *Config.universal_models,
            *Config.recurrent_models,
        ):
            models = {}
            if model_type in (*Config.recurrent_models, *Config.universal_models):
                model_path = next(Config.OUTPUT_RL_MODELS.glob(f"{model_type.__name__}*"))
                model = model_type()
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model.to(Config.device)
                models = dict(
                    (tasks, model)
                    for tasks in range(Config.min_size, Config.n_tasks + 1)
                )
            else:
                for model_path in Config.OUTPUT_RL_MODELS.glob(f"{model_type.__name__}_*"):
                    tasks = int(re.findall(r"_(\d+)", model_path.name)[0])
                    model = model_type(tasks, Config.m_machines)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    model.to(Config.device)
                    models[tasks] = model
            if model_type not in Config.recurrent_models:
                evaluation = getattr(
                    getattr(__import__(f'experiments.{Config.series_model_experiment}_eval'), Config.series_model_experiment + '_eval'),
                    Config.series_model_experiment + '_eval')
            else:
                evaluation = getattr(
                    getattr(__import__(f'experiments.{Config.recurrent_model_experiment}_eval'), Config.recurrent_model_experiment + '_eval'),
                    Config.recurrent_model_experiment + '_eval')
            evaluation(
                Config.eval_iterations,
                models,
                Config.time_constraints,
                Config.beta_constraints,
            )
