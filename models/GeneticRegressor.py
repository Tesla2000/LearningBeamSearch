import random
from itertools import pairwise, starmap

import torch
from torch import nn

from models.abstract.BaseRegressor import BaseRegressor


class GeneticRegressor(BaseRegressor):
    mutations = [
        "INCREASE_HIDDEN_SIZE",
        "DECREASE_HIDDEN_SIZE",
        "ADD_LAYER",
        "DELETE_LAYER",
    ]

    def __init__(
        self, n_tasks: int, m_machines: int, hidden_sizes: tuple[int, ...] = None
    ):
        super().__init__()
        self.m_machines = m_machines
        self.n_tasks = n_tasks
        if not hidden_sizes:
            self.hidden_sizes = (
                (self.n_tasks + 1) * self.m_machines,
                *(random.randint(20, 40) for _ in range(random.randint(0, 3))),
                1,
            )
        else:
            self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList(starmap(nn.Linear, pairwise(self.hidden_sizes)))
        self.flatten = nn.Flatten()
        self.predictions = None
        self.name = type(self).__name__ + '_'.join(map(str, self.hidden_sizes))
        self.correctness_of_predictions = []

    def predict(self, x: torch.Tensor, *args, **kwargs):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            x = self.leaky_relu(x)
        return x

    @classmethod
    def mutate(cls, specimen: tuple[int, ...], mutation: str = None) -> tuple[int, ...]:
        if mutation not in cls.mutations:
            raise ValueError
        if mutation is None:
            mutation = random.choice(cls.mutations)
        hidden_sizes = list(specimen)
        if mutation == "INCREASE_HIDDEN_SIZE":
            if len(hidden_sizes) == 2:
                hidden_sizes.insert(1, 20)
            else:
                hidden_sizes[random.randint(1, len(hidden_sizes) - 2)] += 5
        elif mutation == "DECREASE_HIDDEN_SIZE":
            if len(hidden_sizes) > 2:
                index = random.randint(1, len(hidden_sizes) - 2)
                hidden_sizes[index] -= 5
                if hidden_sizes[index] <= 0:
                    hidden_sizes.pop(index)
        elif mutation == "ADD_LAYER":
            hidden_sizes.insert(-1, 20)
        elif mutation == "DELETE_LAYER":
            if len(hidden_sizes) > 2:
                hidden_sizes.pop(-2)
        return tuple(hidden_sizes)

    def __hash__(self):
        return self.hidden_sizes.__hash__()

    def __eq__(self, other):
        if not isinstance(other, GeneticRegressor):
            return False
        return other.hidden_sizes == self.hidden_sizes

    def __repr__(self):
        return self.name
