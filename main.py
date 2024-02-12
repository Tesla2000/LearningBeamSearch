import random
from itertools import product

import numpy as np
import torch

from Config import Config
from model_training.train_regressor import train_regressor
from regression_models import MultilayerPerceptron
from regression_models.Perceptron import Perceptron

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    m_machines = 25
    for model_type, n_tasks in product(
        (
            # ConvRegressor,
            Perceptron,
            MultilayerPerceptron,
            # WideMultilayerPerceptron,
            # WideConvRegressor,
        ),
        range(Config.min_model_size, Config.n_tasks),
    ):
        model = model_type(n_tasks=n_tasks, m_machines=m_machines)
        train_regressor(model, n_tasks, m_machines)
