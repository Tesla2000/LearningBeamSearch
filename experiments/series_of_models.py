from collections import deque
from itertools import count
from statistics import fmean
from time import time
from typing import IO

import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from Config import Config
from beam_search.Tree import Tree
from model_training.RLDataset import RLDataset
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from model_training.save_models import save_models
from regression_models.GeneticRegressor import GeneticRegressor


def series_of_models(
    n_tasks: int,
    m_machines: int,
    min_size: int,
    generator: RandomNumberGenerator,
    models: dict[int, nn.Module] = None,
    output_file: IO = None,
    **_,
):
    training_buffers = dict(
        (key, deque(maxlen=Config.train_buffer_size)) for key in models
    )
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    batch_size = 32
    optimizers = dict(
        (
            model,
            optim.Adam(
                (
                    model.best_model if isinstance(model, GeneticRegressor) else model
                ).parameters(),
                lr=getattr(model, "learning_rate", 1e-5),
            ),
        )
        for model in models.values()
    )
    schedulers = dict(
        (optimizer, ExponentialLR(optimizer, Config.gamma))
        for optimizer in optimizers.values()
    )
    start = time()
    for epoch in count(1):
        if start + Config.train_time < time():
            break
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
        buffered_results.append(label.item())
        results.append(fmean(buffered_results))
        output_file.write(f"{int(time() - start)},{results[-1]:.2f}\n")
        print(epoch, results[-1])
        for tasks, model in models.items():
            model.train()
            dataset = RLDataset(training_buffers[tasks])
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
            Config.beta[tasks] *= Config.beta_attrition
        if epoch % Config.save_interval == 0:
            save_models(models)
    save_models(models)
