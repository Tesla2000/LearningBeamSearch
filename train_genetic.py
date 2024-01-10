import os
import random
import re
from itertools import chain, pairwise, islice, count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Config import Config
from genetic_models.optimalization_functions import init_state, conv_to_two_d, calc_result
from genetic_models.population_functions import init_population, roulette_choose, mutate, cross_over


def run_genetic(n_tasks: int, n_machines: int, batch_size: int, elitism: int, load_latest: bool = True):
    base_input, targets, best_result = init_state(
        n_tasks, n_machines, batch_size=batch_size, calc_best=True
    )
    start_generation = 0
    if load_latest:
        path = max(Config.POPULATIONS.glob(f'{n_tasks}_{n_machines}_*.txt'),
                   key=lambda path: int(re.findall(r'\d+', path.name)[-1]))
        start_generation = int(re.findall(r'\d+', path.name)[-1])
        population = list(map(np.array, eval(path.read_text())))
    else:
        population = init_population(base_input, pop_size=100, prob_1=.002)
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
        for path in Path(f'{Config.POPULATIONS}').glob(f'{n_tasks}_{n_machines}_*'):
            os.remove(path)
        Path(f'{Config.POPULATIONS}/{n_tasks}_{n_machines}_{best_result:.3f}_{generation}.txt').write_text(
            f"[{','.join(str(list(specimen)) for specimen in population)}]")


def main():
    n_machines = 25
    elitism = 2
    batch_size = 10_000
    load_latest = False
    for n_tasks in range(3, 11):
        run_genetic(n_tasks, n_machines, batch_size, elitism, load_latest)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
