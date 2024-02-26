import random
from collections import deque
from itertools import pairwise, starmap
from typing import Self

import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split

from Config import Config
from regression_models.abstract.BaseRegressor import BaseRegressor


class _GeneticModel(BaseRegressor):
    mutations = [
        'INCREASE_HIDDEN_SIZE',
        'DECREASE_HIDDEN_SIZE',
        'ADD_LAYER',
        'DELETE_LAYER',
    ]

    def __init__(self, n_tasks: int, m_machines: int, hidden_sizes: list[int] = None):
        super().__init__()
        self.m_machines = m_machines
        self.n_tasks = n_tasks
        if not hidden_sizes:
            self.hidden_sizes = [(n_tasks + 1) * m_machines, 1]
        else:
            self.hidden_sizes = hidden_sizes
        self.layers = list(starmap(nn.Linear, pairwise(self.hidden_sizes)))

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

    def mutate(self, mutation: str = None) -> Self:
        if mutation not in self.mutations:
            raise ValueError
        if mutation is None:
            mutation = random.choice(self.mutations)
        hidden_sizes = self.hidden_sizes.copy()
        if mutation == 'INCREASE_HIDDEN_SIZE':
            if len(hidden_sizes) == 2:
                hidden_sizes.insert(1, 5)
            else:
                hidden_sizes[random.randint(1, len(hidden_sizes) - 2)] += 5
        elif mutation == 'DECREASE_HIDDEN_SIZE':
            if len(hidden_sizes) > 2:
                index = random.randint(1, len(hidden_sizes) - 2)
                hidden_sizes[index] -= 5
                if hidden_sizes[index] <= 0:
                    hidden_sizes.pop(index)
        elif mutation == 'ADD_LAYER':
            hidden_sizes.insert(-2, 5)
        elif mutation == 'DELETE_LAYER':
            if len(hidden_sizes) > 2:
                hidden_sizes.pop(-2)
        return _GeneticModel(self.n_tasks, self.m_machines, hidden_sizes)


class GeneticRegressor:
    def __init__(self, n_tasks: int, m_machines: int, n_models: int, **_):
        self.n_tasks = n_tasks
        self.m_machines = m_machines
        self.population = deque(
            (_GeneticModel(n_tasks, m_machines, self._get_random_architecture()) for _ in range(n_models)),
            maxlen=n_models)
        self.best_model = random.choice(self.population)

    def __call__(self, x):
        return self.best_model(x)

    def train_generic(self, dataset, optimizer, criterion, batch_size):
        train_dataset, val_dataset = random_split(dataset, [.8, .2])
        train_loader = DataLoader(train_dataset, batch_size=min(Config.max_status_length, batch_size))
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        losses = []
        for model in self.population:
            model.train()
            for _ in range(Config.gen_train_epochs):
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs.float()).unsqueeze(-1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            model.eval()
            inputs, labels = next(iter(val_loader))
            outputs = model(inputs.float()).unsqueeze(-1)
            losses.append(criterion(outputs, labels))
        self.best_model = self.population[np.argmin(losses)]
        self.population.append(self._mutate(self.best_model))

    def _mutate(self, model: _GeneticModel) -> _GeneticModel:
        mutation = random.choice(model.mutations)
        return model.mutate(mutation)

    def _get_random_architecture(self):
        return [(self.n_tasks + 1) * self.m_machines, *(random.randint(1, 50) for _ in range(random.randint(0, 2))), 1]

    def train(self):
        pass

    def eval(self):
        pass
