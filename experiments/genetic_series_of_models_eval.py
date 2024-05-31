import random
import re
from collections import defaultdict
from itertools import product
from statistics import fmean

import numpy as np
import torch

from Config import Config
from beam_search.GeneticTree import GeneticTree
from experiments._calc_beta import _calc_beta
from experiments._get_beta_genetic_models import _get_beta_genetic_models
from experiments.save_results import save_results
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from models.GeneticRegressor import GeneticRegressor


def genetic_series_of_models_eval(
        iterations: int,
        models_lists: dict[int, dict[GeneticRegressor, float]],
        time_constraints: list[int],
        beta_constraints: list[int],
        **_,
):
    for time_constraint in time_constraints:
        beta = _calc_beta(models_lists, time_constraint, genetic=True)
        beta_dict, filtered_models = _get_beta_genetic_models(models_lists, beta)
        results = []
        torch.manual_seed(Config.evaluation_seed)
        torch.cuda.manual_seed(Config.evaluation_seed)
        np.random.seed(Config.evaluation_seed)
        random.seed(Config.evaluation_seed)
        generator = RandomNumberGenerator(Config.evaluation_seed)
        perceptron_predictions_correct = defaultdict(int)
        for i in range(iterations):
            working_time_matrix = generate_taillard(generator)
            tree = GeneticTree(working_time_matrix, filtered_models)
            bs_results = tree.beam_search(beta_dict)
            for tasks in range(min(filtered_models.keys()), Config.n_tasks):
                try:
                    model = next(filter(lambda model: len(re.findall(r'\d+', model.name)) == 2, filtered_models[tasks]))
                except StopIteration:
                    continue
                for (task_order, state), model_prediction in product(bs_results, model.predictions):
                    prediction_correct = np.array_equal(model_prediction, task_order[:len(model_prediction)])
                    perceptron_predictions_correct[tasks] += prediction_correct
                    if prediction_correct:
                        break
            _, state = bs_results[0]
            results.append(state[-1, -1])
            print(i, GeneticRegressor.__name__, fmean(results))
            print(perceptron_predictions_correct)
        save_results(GeneticRegressor, Config.n_tasks, False, results)
        Config.OUTPUT_RL_RESULTS.joinpath(f"{Config.n_tasks}/{Config.m_machines}/time/{GeneticRegressor.__name__}_{beta}_").write_text(str(results))


