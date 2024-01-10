import random
from math import comb
from typing import Sequence

import numpy as np


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