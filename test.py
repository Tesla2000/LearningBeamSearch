import time
from pathlib import Path
from typing import Type

import numpy as np
import torch
from torch import nn

from Config import Config
from beam_search.Tree import Tree
from beam_search.generator import RandomNumberGenerator
from classification_models.BaseClassifier import BaseClassifier
from classification_models.MinClassifier import MinClassifier
from regression_models import ConvRegressor, GRURegressor, MultilayerPerceptron
from regression_models.WideConvRegressor import WideConvRegressor
from regression_models.abstract.BaseRegressor import BaseRegressor


def test(models: dict[int, BaseClassifier], n_tasks, n_machines):
    generator = RandomNumberGenerator()
    working_time_matrix = np.array(
        [[generator.nextInt(1, 255) for _ in range(n_machines)] for _ in range(n_tasks)]
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
    regressor_type: Type[BaseRegressor] = WideConvRegressor
    classifier_type: Type[BaseClassifier] = MinClassifier
    max_tasks = 10
    n_machines = 25
    min_tasks = 3

    classifier_models = {}
    # if model_type == GRURegressor:
    #     model_parameters_path = tuple(
    #         Path(Config.OUTPUT_REGRESSION_MODELS).glob(f"{model_type.__name__}_*")
    #     )
    #     model = model_type(**locals())
    #     model.load_state_dict(torch.load(model_parameters_path[0]))
    #     model.eval()
    #     for rows in range(2, n_tasks):
    #         models[rows] = model
    # else:
    for n_tasks in range(min_tasks, max_tasks + 1):
        regressor_parameters_path = tuple(
            Path(Config.OUTPUT_REGRESSION_MODELS).glob(f"{regressor_type.__name__}_{n_tasks}*")
        )
        classifier_parameters_path = tuple(
            Path(Config.OUTPUT_CLASSIFIER_MODELS).glob(f"{classifier_type.__name__}_{regressor_type.__name__}_{n_tasks}*")
        )
        if not regressor_parameters_path or not classifier_parameters_path:
            continue
        regressor = regressor_type(n_tasks, n_machines)
        regressor.load_state_dict(torch.load(regressor_parameters_path[0]))
        regressor.eval()
        classifier = classifier_type(regressor)
        classifier.load_state_dict(torch.load(classifier_parameters_path[0]))
        classifier.eval()
        classifier_models[n_tasks] = classifier
    test(classifier_models, max_tasks, n_machines)


if __name__ == "__main__":
    main()
