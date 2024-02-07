import random
from itertools import product

import numpy as np
import torch

from model_training.train_regressor import train_regressor
from regression_models.Perceptron import Perceptron

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    n_machines = 25
    for model_type, n_tasks in product(
        (
            # ConvRegressor,
            # MultilayerPerceptron,
            # partial(MultilayerPerceptron, hidden_size=512),
            # SumRegressor,
            Perceptron,
            # WideMultilayerPerceptron,
            # WideConvRegressor,
        ),
        range(3, 11),
    ):
        model = model_type(n_tasks=n_tasks, n_machines=n_machines)
        train_regressor(model, n_tasks, n_machines)
