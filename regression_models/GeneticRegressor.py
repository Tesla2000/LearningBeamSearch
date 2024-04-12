import random
from itertools import pairwise, starmap, count
from sqlite3 import OperationalError

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from Config import Config
from regression_models.abstract.BaseRegressor import BaseRegressor


class _GeneticModel(BaseRegressor):
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
            self.hidden_sizes = ((n_tasks + 1) * m_machines, 1)
        else:
            self.hidden_sizes = hidden_sizes
        self.layers = nn.ModuleList(starmap(nn.Linear, pairwise(self.hidden_sizes)))
        self.flatten = nn.Flatten()

    def predict(self, x):
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
                hidden_sizes.insert(1, 5)
            else:
                hidden_sizes[random.randint(1, len(hidden_sizes) - 2)] += 5
        elif mutation == "DECREASE_HIDDEN_SIZE":
            if len(hidden_sizes) > 2:
                index = random.randint(1, len(hidden_sizes) - 2)
                hidden_sizes[index] -= 5
                if hidden_sizes[index] <= 0:
                    hidden_sizes.pop(index)
        elif mutation == "ADD_LAYER":
            hidden_sizes.insert(-1, 5)
        elif mutation == "DELETE_LAYER":
            if len(hidden_sizes) > 2:
                hidden_sizes.pop(-2)
        return tuple(hidden_sizes)


class GeneticRegressor:
    batch_size: int = 32

    def __init__(self, n_tasks: int, m_machines: int, **_):
        self.n_tasks = n_tasks
        self.m_machines = m_machines
        self.device = Config.device
        self.population = list(
            self._get_random_architecture() for _ in range(Config.n_genetic_models)
        )
        self.pareto: dict[tuple[int], tuple] = {}
        self.best_model = _GeneticModel(
            n_tasks, m_machines, ((self.n_tasks + 1) * self.m_machines, 1)
        )
        self.best_model.to(self.device)
        self.produce_data = True

    def to(self, device):
        self.device = device

    def __call__(self, x):
        return self.best_model(x)

    def train_generic(self, dataset, criterion):
        try:
            if len(dataset) < 5:
                return
        except OperationalError:
            return
        for hidden_sizes in tuple(self.pareto.keys()):
            if random.random() > Config.pareto_retrain_rate:
                continue
            del self.pareto[hidden_sizes]
            self._retrain_hidden_sizes(hidden_sizes, criterion, dataset)
        self.population = list(map(self._mutate, self.population))
        for hidden_sizes in (
            *random.sample(
                tuple(map(self._mutate, self.pareto.keys())),
                k=min(len(self.pareto.keys()), Config.n_pareto_samples),
            ),
            *random.sample(self.population, k=Config.n_population_samples),
        ):
            self._retrain_hidden_sizes(hidden_sizes, criterion, dataset)
        self.best_model = self._retrain_hidden_sizes(
            random.choice(tuple(self.pareto.keys())), criterion, dataset
        )
        self.best_model.to(self.device)

    def _mutate(self, specimen: tuple[int, ...]) -> tuple[int, ...]:
        mutation = random.choice(_GeneticModel.mutations)
        return _GeneticModel.mutate(specimen, mutation)

    def _get_random_architecture(self):
        return (
            (self.n_tasks + 1) * self.m_machines,
            *(random.randint(1, 50) for _ in range(random.randint(0, 1))),
            1,
        )

    def _retrain_hidden_sizes(
        self, hidden_sizes, criterion, dataset
    ) -> _GeneticModel:
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(
            train_dataset, batch_size=min(Config.max_status_length, self.batch_size)
        )
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        model = _GeneticModel(self.n_tasks, self.m_machines, hidden_sizes)
        optimizer = optim.Adam(
            model.parameters(), lr=model.learning_rate
        )
        model.to(self.device)
        prev_loss = float('inf')
        for epoch in count():
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.unsqueeze(-1).float())
                loss.backward()
                optimizer.step()
            model.eval()
            inputs, labels = next(iter(val_loader))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs.float())
            number_of_weights = sum(p.numel() for p in model.parameters())
            loss = criterion(outputs, labels.unsqueeze(-1)).item()
            if loss > prev_loss:
                break
            prev_loss = loss
        self._add_to_pareto(hidden_sizes, number_of_weights, loss)
        return model

    def _add_to_pareto(self, hidden_sizes, number_of_weights, loss):
        for key, (pareto_n_weights, pareto_loss) in tuple(self.pareto.items()):
            if number_of_weights <= pareto_n_weights and loss <= pareto_loss:
                del self.pareto[key]
            elif number_of_weights >= pareto_n_weights and loss >= pareto_loss:
                break
        else:
            self.pareto[hidden_sizes] = (number_of_weights, loss)
