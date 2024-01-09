import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset
from regression_models.NEAT import NEAT, NEATLearningBeamSearchTrainer
from regression_models.Perceptron import Perceptron


def train_neat(model: NEAT, n_tasks: int, n_machines: int):
    batch_size = 1000

    # criterion = nn.MSELoss()
    def criterion(output, target):
        return (output - target).abs()

    neat_trainer = NEATLearningBeamSearchTrainer(model, criterion)
    data_file = Path(
        f"data_generation/untitled/training_data_regression/{n_tasks}_{n_machines}.txt"
    ).open()
    data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=n_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    best_winner = None
    best_score = -float("inf")
    winner_net = None
    try:
        for index, (inputs, targets) in enumerate(train_loader):
            if winner_net:
                score = neat_trainer.score(winner_net, inputs, targets,
                                           # verbose=True
                                           )
                print(index, score / batch_size)
                if score > best_score:
                    best_score = score
                    best_winner = winner
            winner, winner_net = neat_trainer.train(model, inputs, targets, n=1)
    except ValueError:
        pass
    finally:
        model.save_winner(
            f"{Config.OUTPUT_REGRESSION_MODELS}/NEAT_{n_tasks}_{n_machines}.pkl", best_winner
        )


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    n_machines = 25
    for n_tasks in range(3, 11):
        model_path = next(
            Config.OUTPUT_REGRESSION_MODELS.glob(
                f"{Perceptron.__name__}_{n_tasks}_{n_machines}*"
            )
        )
        model = NEAT(
            n_tasks,
            n_machines,
            initial_weights=torch.load(model_path),
            initial_fitness=float(re.findall(r'[\d\.]+', model_path.name)[-1][:-1])
        )
        train_neat(model, n_tasks, n_machines)


if __name__ == "__main__":
    main()
