import random
from collections import deque
from itertools import pairwise, starmap

import numpy as np
from torch import nn, optim
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

    def __init__(self, n_tasks: int, m_machines: int, hidden_sizes: tuple[int, ...] = None):
        super().__init__()
        self.m_machines = m_machines
        self.n_tasks = n_tasks
        if not hidden_sizes:
            self.hidden_sizes = ((n_tasks + 1) * m_machines, 1)
        else:
            self.hidden_sizes = hidden_sizes
        self.layers = tuple(starmap(nn.Linear, pairwise(self.hidden_sizes)))
        tuple(setattr(self, f'l{i}', layer) for i, layer in enumerate(self.layers))
        self.flatten = nn.Flatten()

    def predict(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

    @classmethod
    def mutate(cls, specimen: tuple[int, ...], mutation: str = None) -> tuple[int, ...]:
        if mutation not in cls.mutations:
            raise ValueError
        if mutation is None:
            mutation = random.choice(cls.mutations)
        hidden_sizes = list(specimen)
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
            hidden_sizes.insert(-1, 5)
        elif mutation == 'DELETE_LAYER':
            if len(hidden_sizes) > 2:
                hidden_sizes.pop(-2)
        return tuple(hidden_sizes)


class GeneticRegressor:
    def __init__(self, n_tasks: int, m_machines: int, **_):
        self.n_tasks = n_tasks
        self.m_machines = m_machines
        self.device = Config.device
        self.population = deque(
            (self._get_random_architecture() for _ in range(Config.n_genetic_models)),
            maxlen=Config.n_genetic_models)
        self.best_model = _GeneticModel(n_tasks, m_machines, random.choice(self.population))
        self.best_model.to(self.device)
        self._results = {}

    def to(self, device):
        self.device = device

    def __call__(self, x):
        return self.best_model(x)

    def train_generic(self, dataset, criterion, batch_size):
        train_dataset, val_dataset = random_split(dataset, [.8, .2])
        train_loader = DataLoader(train_dataset, batch_size=min(Config.max_status_length, batch_size))
        if not len(val_dataset):
            return
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        losses = []
        for hidden_sizes in random.sample(self.population, Config.n_genetic_samples):
            if hidden_sizes in self._results and random.random() > Config.retrain_rate:
                losses.append(self._results[hidden_sizes])
                continue
            model = _GeneticModel(self.n_tasks, self.m_machines, hidden_sizes)
            optimizer = optim.Adam(model.parameters(), lr=getattr(model, 'learning_rate', 1e-5))
            model.to(self.device)
            for _ in range(Config.gen_train_epochs):
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels.unsqueeze(-1).float())
                    loss.backward()
                    optimizer.step()
            model.eval()
            inputs, labels = next(iter(val_loader))
            outputs = model(inputs.float())
            losses.append(criterion(outputs, labels.unsqueeze(-1)).item() + Config.size_penalty * sum(p.numel() for p in model.parameters()))
            if losses[-1] == min(losses):
                self.best_model = model
            self._results[hidden_sizes] = losses[-1]
        best_specimen = self.population[np.argmin(losses)]
        self.population.append(self._mutate(best_specimen))

    def _mutate(self, specimen: tuple[int, ...]) -> tuple[int, ...]:
        mutation = random.choice(_GeneticModel.mutations)
        return _GeneticModel.mutate(specimen, mutation)

    def _get_random_architecture(self):
        return (self.n_tasks + 1) * self.m_machines, *(random.randint(1, 50) for _ in range(random.randint(0, 2))), 1

    def train(self):
        pass

    def eval(self):
        pass
