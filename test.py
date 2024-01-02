import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from beam_search.generator import RandomNumberGenerator
from regression_models import ConvRegressor, GRURegressor, MultilayerPerceptron


def test(models: dict[int, nn.Module], n_tasks, n_machines):
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 99) for _ in range(n_machines)] for _ in range(n_tasks)]
    )
    model_tree = Tree(working_time_matrix, models)
    branch_and_bound_tree = Tree(working_time_matrix, {})
    start = time.time()
    model_value = model_tree.eval_with_model().value
    model_time = time.time() - start
    print("Model", model_value, model_time)

    start = time.time()
    b_b_value = branch_and_bound_tree.eval_with_model().value
    b_b_time = time.time() - start
    print("Branch And Bound", b_b_value, b_b_time)


def main():
    model_type = GRURegressor
    n_tasks = 7
    n_machines = 10

    models = {}
    if model_type == GRURegressor:
        model_parameters_path = tuple(
            Path(Config.OUTPUT_REGRESSION_MODELS).glob(f"{model_type.__name__}_*")
        )
        model = model_type(**locals())
        model.load_state_dict(torch.load(model_parameters_path[0]))
        model.eval()
        for rows in range(2, n_tasks):
            models[rows] = model
    else:
        for rows in range(2, n_tasks):
            model_parameters_path = tuple(
                Path(Config.OUTPUT_REGRESSION_MODELS).glob(f"{model_type.__name__}_{rows}*")
            )
            if not model_parameters_path:
                continue
            model = model_type(**locals())
            model.load_state_dict(torch.load(model_parameters_path[0]))
            model.eval()
            models[rows] = model
    test(models, n_tasks, n_machines)


if __name__ == "__main__":
    main()
