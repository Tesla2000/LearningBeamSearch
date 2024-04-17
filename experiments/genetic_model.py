import json
from math import log
from time import time
from typing import IO

import numpy as np
from torch import nn
from tqdm import tqdm

from Config import Config
from beam_search.Tree import Tree
from experiments.series_of_models import series_of_models
from model_training.RLDataset import RLDataset
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from model_training.save_models import save_models
from regression_models.GeneticRegressorCreator import GeneticRegressorCreator
from regression_models.Perceptron import Perceptron


def genetic_model(
    n_tasks: int,
    m_machines: int,
    min_size: int,
    generator: RandomNumberGenerator,
    output_file: IO = None,
    train_time: int = Config.train_time,
    **_,
):
    models = dict(
        (tasks, Perceptron(tasks, Config.m_machines).to(Config.device))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    series_of_models(
        n_tasks=n_tasks,
        m_machines=m_machines,
        min_size=min_size,
        generator=generator,
        models=models,
        output_file=output_file,
        train_time=train_time // 3,
        output_model_path=Config.OUTPUT_GENETIC_MODELS
    )
    start = time()
    training_buffers = dict(
        (key, list()) for key in models
    )
    while start + train_time // 3 > time():
        working_time_matrix = generate_taillard(generator)
        tree = Tree(working_time_matrix, models)
        task_order, state = tree.beam_search(Config.beta)
        for tasks in range(min_size, n_tasks + 1):
            if tasks == n_tasks:
                header = np.zeros((1, m_machines))
            else:
                header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data, axis=0)
            label = state[-1, -1]
            training_buffers[tasks].append((data, label))
    models = dict(
        (tasks, GeneticRegressorCreator(tasks, Config.m_machines))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    start = time()
    criterion = nn.MSELoss()
    while start + train_time // 3 > time():
        for tasks, model in tqdm(models.items(), "Training genetic models"):
            dataset = RLDataset(training_buffers[tasks])
            model.train_generic(dataset, criterion)
    for tasks, model in models.items():
        pareto = model.pareto
        Config.OUTPUT_GENETIC_MODELS.joinpath(f"{tasks}.txt").write_text(str(pareto))
    for tasks, model in models.items():
        pareto = model.pareto
        dataset = RLDataset(training_buffers[tasks])
        best_hidden_size = min(pareto, key=lambda key: log(pareto[key][0], 2) * pareto[key][1])
        models[tasks] = model.retrain_hidden_sizes(best_hidden_size, criterion, dataset)
    save_models(models)
