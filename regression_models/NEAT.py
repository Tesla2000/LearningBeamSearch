import pickle
from functools import partial
from pathlib import Path
from typing import Callable

import neat
from torch import nn, Tensor
from time import time

time_coefficient = .01


class NEAT(nn.Module):
    def __init__(
        self,
        n_tasks: int,
        n_machines: int,
        checkpoint_file: Path | str = None,
        winner_path: Path | str = None,
        initial_weights=None,
    ):
        super().__init__()
        config_path = f"neat_configurations/{n_tasks}_{n_machines}.txt"
        Path(config_path).write_text(
            Path(f"neat_configurations/base_config.txt").read_text().format(
                pop_size=10, num_inputs=(n_tasks + 1) * n_machines)
        )
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        checkpointer = neat.Checkpointer(
            50, filename_prefix=f"neat_checkpoints/{n_tasks}_{n_machines}_"
        )
        if checkpoint_file is None:
            self.population = neat.Population(self.config)
        else:
            self.population = checkpointer.restore_checkpoint(checkpoint_file)
        if initial_weights is not None:
            weights, bias = initial_weights.values()
            for _, specimen in self.population.population.items():
                specimen.nodes[0].bias = float(bias)
                for connection, weight in zip(
                    specimen.connections.values(), map(float, weights[0])
                ):
                    connection.weight = weight
        if winner_path is None:
            self.winner = None
            self.winner_net = None
        else:
            with open(winner_path, "rb") as f:
                self.winner = pickle.load(f)
            self.winner_net = neat.nn.FeedForwardNetwork.create(
                self.winner, self.config
            )

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
            genome.fitness = -sum(
                map(
                    self.scoring_function,
                    map(Tensor, map(net.activate, inputs)),
                    Tensor(targets).reshape(-1, 1),
                )
            ).item()

    def train(self, neat_instance: NEAT, inputs, targets, n: int):
        neat_instance.winner = neat_instance.population.run(
            partial(self._eval_genomes, inputs=inputs, targets=targets), n
        )
        neat_instance.winner_net = neat.nn.FeedForwardNetwork.create(
            neat_instance.winner, neat_instance.config
        )
        return neat_instance.winner, neat_instance.winner_net


class NEATLearningBeamSearchTrainer(NEATTrainer):
    def _eval_genomes(self, population, config, inputs, targets):
        for _, specimen in population:
            net = neat.nn.FeedForwardNetwork.create(specimen, config)
            specimen.fitness = self.score(net, inputs, targets) / len(inputs)

    def score(self, net, inputs, targets, verbose=False):
        def get_output(x):
            min_value = float(x[0, -1])
            min_value += sum(x[1:, -1])
            x = net.activate(x.flatten())[0]
            return (x + min_value).reshape(1)

        start = time()
        result = -sum(
            map(
                self.scoring_function,
                map(get_output, inputs),
                Tensor(targets).reshape(-1, 1),
            )
        ).item()
        time_result = time() + start
        time_result *= time_coefficient / len(inputs)
        if verbose:
            print(result, time_result)
        return result - time_result
