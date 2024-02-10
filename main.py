import random

import numpy as np
import torch

from Config import Config
from model_training.train_rl import train_rl
from regression_models.MultilayerPerceptron import MultilayerPerceptron
from regression_models.Perceptron import Perceptron

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    models = dict(
        (tasks, Config.model_type(tasks, Config.m_machines))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    for model in models.values():
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    fill_strings = {}
    train_rl(
        Config.n_tasks,
        Config.m_machines,
        Config.iterations,
        Config.min_size,
        models,
        Config.MODEL_RESULTS.joinpath(
            f"{Config.model_type.__name__}_{Config.n_tasks}_{Config.m_machines}_{Config.min_size}"
        ).open('w'),
    )
