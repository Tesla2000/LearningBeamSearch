from collections import deque
from itertools import count
from statistics import fmean
from time import time
from typing import IO

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from Config import Config
from beam_search.GeneticTree import GeneticTree
from model_training.RLDataset import RLDataset
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from model_training.save_models import save_models
from models.GeneticRegressor import GeneticRegressor


def genetic_series_of_models(
    n_tasks: int,
    m_machines: int,
    min_size: int,
    generator: RandomNumberGenerator,
    output_file: IO = None,
    train_time: int = Config.train_time,
    **_,
):
    training_buffers = dict(
        (tasks, deque(maxlen=Config.train_buffer_size)) for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    batch_size = 32
    models_lists = dict(
        (tasks, list(set(GeneticRegressor(tasks, Config.m_machines).to(Config.device) for _ in range(Config.n_genetic_models))))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    optimizers = dict(
        (
            model,
            optim.Adam(
                model.parameters(),
                lr=getattr(model, "learning_rate", 1e-5),
            ),
        )
        for models in models_lists.values() for model in models
    )
    schedulers = dict(
        (optimizer, ExponentialLR(optimizer, Config.gamma))
        for optimizer in optimizers.values()
    )
    start = time()
    for epoch in count(1):
        if start + train_time < time():
            break
        working_time_matrix = generate_taillard(generator)
        tree = GeneticTree(working_time_matrix, models_lists)
        result = tree.beam_search(Config.genetic_beta)
        tuple(model.correctness_of_predictions.append(False) for models in models_lists.values() for model in models)
        for task_order, state in result:
            for tasks in range(min_size, n_tasks):
                for model in models_lists[tasks]:
                    # if model.predictions is None:
                    #     continue
                    for model_prediction in model.predictions:
                        prediction_correct = np.array_equal(model_prediction, task_order[:len(model_prediction)])
                        model.correctness_of_predictions[-1] = model.correctness_of_predictions[-1] or prediction_correct
                if tasks == n_tasks:
                    header = np.zeros((1, m_machines))
                else:
                    header = state[-tasks - 1].reshape(1, -1)
                data = working_time_matrix[list(task_order[-tasks:])]
                data = np.append(header, data, axis=0)
                label = state[-1, -1]
                training_buffers[tasks].append((data, label))
        correct_model = tuple(min(filter(lambda model: model.correctness_of_predictions[-1], models_lists[tasks]), key=lambda model: sum(p.numel() for p in model.parameters()))for tasks in range(min_size, n_tasks))
        tuple(model.correctness_of_predictions.__setitem__(-1, False) for models in models_lists.values() for model in models)
        tuple(model.correctness_of_predictions.__setitem__(-1, True) for model in correct_model)
        buffered_results.append(label.item())
        results.append(fmean(buffered_results))
        output_file.write(f"{int(time() - start)},{results[-1]:.2f}\n")
        print(epoch, results[-1])
        for tasks, models in models_lists.items():
            dataset = RLDataset(training_buffers[tasks])
            for model in models:
                model.train()
                optimizer = optimizers[model]
                train_loader = DataLoader(
                    dataset, batch_size=min(Config.max_status_length, batch_size)
                )
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                    labels = labels.float().unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs.float())
                    loss = Config.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                schedulers[optimizer].step()
            if epoch % Config.save_interval == 0:
                save_models(dict((tasks, model) for model in models), Config.OUTPUT_GENETIC_MODELS)
    for tasks, models in models_lists.items():
        save_models(dict((tasks, model) for model in models), Config.OUTPUT_GENETIC_MODELS)
