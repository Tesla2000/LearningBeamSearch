import random
import re
from itertools import chain, pairwise, islice, count, combinations_with_replacement
from math import comb
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
from numpy.linalg import lstsq
from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import Config
from regression_models.RegressionDataset import RegressionDataset


def conv_to_two_d(array: np.array, mask: Sequence[bool]):
    coef_combinations = np.array(tuple(combinations_with_replacement(range(len(array[0]) - 1), 2)))
    coef_combinations = coef_combinations[mask == 1]
    return np.concatenate((array, *tuple(np.multiply(array[:, combo[0]], array[:, combo[1]]).reshape(-1, 1) for combo in coef_combinations)), axis=1)


def init_state(
    n_tasks: int, n_machines: int, batch_size: int = None, calc_best: bool = False
) -> tuple[np.array, np.array, Optional[float]]:
    data_file = Path(
        f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt"
    ).open()
    data_maker = RegressionDataset(
        n_tasks=n_tasks, n_machines=n_machines, data_file=data_file
    )
    if batch_size is None:
        batch_size = len(data_maker)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    inputs, labels = next(iter(train_loader))
    labels = labels.numpy()
    inputs = inputs.flatten(start_dim=1).numpy()
    mod_input = np.empty((inputs.shape[0], inputs.shape[1] + 1))
    mod_input[:, : inputs.shape[1]] = inputs
    del inputs
    best_results = None
    mod_input[:, -1] = np.ones_like(mod_input[:, -1])
    if calc_best:
        best_results = calc_result(mod_input, labels)
    return mod_input, labels, best_results


def calc_result(input, target):
    w = lstsq(input, target, rcond=-1)[0]
    output = input.dot(w)
    return np.mean(np.abs(target - output))


def cross_over(parents: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
    speciment_1, speciment_2 = parents
    point_1, point_2 = sorted(random.sample(range(len(speciment_1)), 2))
    return (
        np.concatenate(
            (speciment_1[:point_1], speciment_2[point_1:point_2], speciment_1[point_2:])
        ),
        np.concatenate(
            (speciment_2[:point_1], speciment_1[point_1:point_2], speciment_2[point_2:])
        ),
    )


def mutate(specimen: np.array) -> np.array:
    position = random.choice(range(len(specimen)))
    specimen[position] += 1
    specimen[position] %= 2
    return specimen


def roulette_choose(
    population: Sequence[np.array], fitness: Sequence[float], elitism: int
) -> np.array:
    fitness = np.array(fitness)
    elite = [population[index] for index in np.argsort(fitness)[:elitism]]
    fitness = 1 / fitness
    fitness -= np.min(fitness)
    return elite + random.choices(population, fitness, k=len(population) - elitism)


def init_population(
    base_input: np.array, pop_size: int = 10, prob_1: float = 0.002
) -> tuple:
    return tuple(
        np.random.choice(
            [0, 1], size=len(base_input[0]) - 1 + comb(len(base_input[0]) - 1, 2), p=[1 - prob_1, prob_1]
        )
        for _ in range(pop_size)
    )


def main():
    n_machines = 25
    elitism = 2
    batch_size = 10_000
    load_latest = False
    for n_tasks in range(3, 11):
        base_input, targets, best_result = init_state(
            n_tasks, n_machines, batch_size=batch_size
        )
        start_generation = 0
        if load_latest:
            path = max(Config.POPULATIONS.glob(f'{n_tasks}_{n_machines}_*.txt'), key=lambda path: int(re.findall(r'\d+', path.name)[-1]))
            start_generation = int(re.findall(r'\d+', path.name)[-1])
            population = list(map(np.array, eval(path.read_text())))
        else:
            population = init_population(base_input, prob_1=.01)
        print(best_result)
        for generation in tqdm(count(start_generation), "Evaluating generation..."):
            fitness = []
            for specimen in population:
                mod_input = conv_to_two_d(base_input, mask=specimen)
                result = calc_result(mod_input, targets)
                fitness.append(result)
                if best_result is None:
                    best_result = result
                    print(best_result)
                if best_result > result:
                    best_result = result
                    print(best_result, sum(specimen))
            population = roulette_choose(population, fitness, elitism)
            population = population[:elitism] + list(
                map(
                    mutate,
                    chain.from_iterable(
                        map(
                            cross_over,
                            islice(pairwise(population[elitism:]), 0, None, 2),
                        )
                    ),
                )
            )
            Path(f'{Config.POPULATIONS}/{n_tasks}_{n_machines}_{best_result:.3f}_{generation}.txt').write_text(f"[{','.join(str(list(specimen)) for specimen in population)}]")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()