from torch import nn

from Config import Config
from model_training.RegressionDataset import (
    RegressionDataset,
)
from regression_models.GeneticRegressor import GeneticRegressor


def train_genetic(m_machines: int):
    models = dict(
        (tasks, GeneticRegressor(tasks, Config.m_machines))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    criterion = nn.MSELoss()
    while True:
        for tasks, model in models.items():
            dataset = RegressionDataset(n_tasks=tasks, m_machines=m_machines)
            model.train_generic(dataset, criterion)
            break


if __name__ == '__main__':
    train_genetic(Config.m_machines)
