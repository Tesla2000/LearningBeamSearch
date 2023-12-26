import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from beam_search.generator import RandomNumberGenerator
from ml_models import ConvModel, LSTMModel, DenseModel


def test(models: nn.Module):
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(n_machines)]
    )
    tree = Tree(working_time_matrix, models)
    start = time.time()
    model_value = tree.eval_with_model()
    model_time = time.time() - start

    start = time.time()
    b_b_value = tree.branch_and_bound()
    b_b_time = time.time() - start


if __name__ == "__main__":
    model_type = DenseModel
    n_tasks = 7
    n_machines = 10
    models = {}
    for rows in range(2, n_tasks):
        model_parameters_path = tuple(Path(Config.OUTPUT_MODELS).glob(f'{model_type.__name__}_{rows}*'))
        if not model_parameters_path:
            continue
        model = model_type(rows, n_machines).load_state_dict(torch.load(model_parameters_path[0]))
        models[rows] = model
    test(models)
