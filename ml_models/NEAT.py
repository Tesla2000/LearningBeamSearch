import pickle
from functools import partial
from pathlib import Path
from typing import Callable

import neat
from torch import nn, Tensor


class NEAT(nn.Module):
    def __init__(self, n_tasks: int, n_machines: int, checkpoint_file: Path | str = None,
                 winner_path: Path | str = None):
        super().__init__()
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            f"neat_configurations/{n_tasks}_{n_machines}.txt",
        )

        stats = neat.StatisticsReporter()
        checkpointer = neat.Checkpointer(50, filename_prefix=f'neat_checkpoints/{n_tasks}_{n_machines}')
        if checkpoint_file is None:
            self.population = neat.Population(self.config)
        else:
            self.population = checkpointer.restore_checkpoint(checkpoint_file)
        if winner_path is None:
            self.winner = None
            self.winner_net = None
        else:
            with open(winner_path, 'rb') as f:
                self.winner = pickle.load(f)
            self.winner_net = neat.nn.FeedForwardNetwork.create(self.winner, self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(stats)
        self.population.add_reporter(checkpointer)

    def forward(self, x):
        return self.winner_net.activate(x)

    def save_winner(self, winner_path: Path | str, winner=None):
        with open(winner_path, "wb") as f:
            pickle.dump(winner or self.winner, f)


class NEATTrainer:
    def __init__(self, scoring_function: Callable[[Tensor, Tensor], Tensor]):
        self.scoring_function = scoring_function

    def _eval_genomes(self, genomes, config, inputs, targets):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = -sum(map(self.scoring_function, map(Tensor, map(net.activate, inputs)),
                                      Tensor(targets).reshape(-1, 1))).item()

    def train(self, neat_instance: NEAT, inputs, targets, n: int):
        neat_instance.winner = neat_instance.population.run(
            partial(self._eval_genomes, inputs=inputs, targets=targets), n
        )
        neat_instance.winner_net = neat.nn.FeedForwardNetwork.create(neat_instance.winner, neat_instance.config)
        return neat_instance.winner, neat_instance.winner_net


class NEATLearningBeamSearchTrainer(NEATTrainer):
    def _eval_genomes(self, genomes, config, inputs, targets):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.score(net, inputs, targets)

    def score(self, net, inputs, targets):
        def get_output(x):
            nonlocal net
            min_value = x[0, -1]
            min_value += sum(x[1:, -1])
            x = net.activate(x.flatten())[0]
            return (x + min_value).reshape(1)

        return -sum(
            map(
                self.scoring_function,
                map(get_output, inputs),
                Tensor(targets).reshape(-1, 1),
            )
        ).item()
