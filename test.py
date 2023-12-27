import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from beam_search.generator import RandomNumberGenerator
from ml_models import ConvModel, LSTMModel, DenseModel


def test(models: dict[int, nn.Module]):
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 99) for _ in range(n_machines)] for _ in range(n_tasks)]
    )
    tree = Tree(working_time_matrix, models)
    start = time.time()
    model_value = tree.eval_with_model().value
    model_time = time.time() - start
    print("Model", model_value, model_time)

    start = time.time()
    b_b_value = tree.branch_and_bound().value
    b_b_time = time.time() - start
    print("Branch And Bound", b_b_value, b_b_time)


def test_recurrent(models: dict[int, nn.Module]):
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 99) for _ in range(n_machines)] for _ in range(n_tasks)]
    )
    tree = Tree(working_time_matrix, models)
    start = time.time()
    model_value = tree.eval_recurrent_model_with_model().value
    model_time = time.time() - start
    print("Model", model_value, model_time)

    start = time.time()
    b_b_value = tree.branch_and_bound().value
    b_b_time = time.time() - start
    print("Branch And Bound", b_b_value, b_b_time)


if __name__ == "__main__":
    model_type = LSTMModel
    n_tasks = 7
    n_machines = 10
    models = {}
    if model_type == LSTMModel:
        model_parameters_path = tuple(Path(Config.OUTPUT_MODELS).glob(f'{model_type.__name__}_*'))
        model = model_type(**locals())
        model.load_state_dict(torch.load(model_parameters_path[0]))
        model.eval()
        models[None] = model
        test_recurrent(models)
    else:
        for rows in range(2, n_tasks):
            model_parameters_path = tuple(Path(Config.OUTPUT_MODELS).glob(f'{model_type.__name__}_{rows}*'))
            if not model_parameters_path:
                continue
            model = model_type(**locals())
            model.load_state_dict(torch.load(model_parameters_path[0]))
            model.eval()
            models[rows] = model
        test(models)

