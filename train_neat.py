from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset
from regression_models.NEAT import NEAT, NEATLearningBeamSearchTrainer
from regression_models.Perceptron import Perceptron


def train_neat(model: NEAT, n_tasks: int, n_machines: int):
    batch_size = 10

    # criterion = nn.MSELoss()
    def criterion(output, target):
        return (output - target).abs()

    neat_trainer = NEATLearningBeamSearchTrainer(criterion)
    data_file = Path(
        f"data_generation/untitled/training_data/{n_tasks}_{n_machines}.txt"
    ).open()
    data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=n_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    average = 100
    patience = 1000
    scoring_inputs = deque(maxlen=average * batch_size)
    scoring_targets = deque(maxlen=average * batch_size)
    winner_nets = deque(maxlen=average)
    winners = deque(maxlen=average)
    best_winner = None
    best_score = -float("inf")
    best_score_index = None
    try:
        for index, (inputs, targets) in enumerate(train_loader):
            scoring_inputs.extend(inputs)
            scoring_targets.extend(targets)
            winner, winner_net = neat_trainer.train(model, inputs, targets, n=1)
            winner_nets.append(winner_net)
            winners.append(winner_net)
            if len(winner_nets) != average:
                continue
            score = neat_trainer.score(winner_nets[0], scoring_inputs, scoring_targets)
            print(index, score / len(scoring_inputs))
            if score > best_score:
                best_score = score
                best_score_index = index
                best_winner = winners[0]
            if best_score_index + patience < index:
                break

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
        model = NEAT(
            n_tasks,
            n_machines,
            initial_weights=torch.load(
                next(
                    Config.OUTPUT_REGRESSION_MODELS.glob(
                        f"{Perceptron.__name__}_{n_tasks}_{n_machines}*"
                    )
                )
            ),
        )
        train_neat(model, n_tasks, n_machines)


if __name__ == "__main__":
    main()
