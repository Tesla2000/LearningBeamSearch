import re
from itertools import count
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Config import Config
from model_training.RegressionDataset import RegressionDataset
from models.NEAT import NEAT, NEATLearningBeamSearchTrainer
from models.Perceptron import Perceptron


def train_neat(model: NEAT, n_tasks: int, m_machines: int):
    # criterion = nn.MSELoss()
    def criterion(output, target):
        return (output - target).abs()

    neat_trainer = NEATLearningBeamSearchTrainer(model, criterion)
    batch_size = 1000
    # data_file = Path(
    #     f"data_generation/untitled/training_data_regression/{n_tasks}_{m_machines}.txt"
    # ).open()
    # data_maker = RegressionDataset(n_tasks=n_tasks, m_machines=m_machines, data_file=data_file)
    # train_loader = DataLoader(data_maker, batch_size=batch_size)
    best_winner = None
    best_score = -float("inf")
    winner_net = None
    for index in count():
        data_file = Path(
            f"data_generation/untitled/training_data_regression/{n_tasks}_{m_machines}.txt"
        ).open()
        data_maker = RegressionDataset(
            n_tasks=n_tasks, m_machines=m_machines, data_file=data_file
        )
        train_loader = DataLoader(data_maker, batch_size=batch_size)
        inputs, targets = next(iter(train_loader))
        winner, winner_net = neat_trainer.train(model, inputs, targets, n=1)
        score = winner.fitness
        if score > best_score:
            best_score = score
            best_winner = winner
        print(index, score)
    # try:
    #     for index, (inputs, targets) in enumerate(train_loader):
    #         if winner_net:
    #             score = neat_trainer.score(winner_net, inputs, targets,
    #                                        # verbose=True
    #                                        )
    #             print(index, score / batch_size)
    #             if score > best_score:
    #                 best_score = score
    #                 best_winner = winner
    #         winner, winner_net = neat_trainer.train(model, inputs, targets, n=1)
    # except ValueError:
    #     pass
    # finally:
    #     model.save_winner(
    #         f"{Config.OUTPUT_REGRESSION_MODELS}/NEAT_{n_tasks}_{m_machines}.pkl", best_winner
    #     )


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    m_machines = 25
    for n_tasks in range(3, 11):
        model_path = next(
            Config.OUTPUT_REGRESSION_MODELS.glob(
                f"{Perceptron.__name__}_{n_tasks}_{m_machines}*"
            )
        )
        model = NEAT(
            n_tasks,
            m_machines,
            pop_size=10,
            initial_weights=torch.load(model_path),
            initial_fitness=float(re.findall(r"[\d\.]+", model_path.name)[-1][:-1]),
        )
        train_neat(model, n_tasks, m_machines)


if __name__ == "__main__":
    main()
