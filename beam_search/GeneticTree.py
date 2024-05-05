from itertools import filterfalse, permutations, product, chain
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from Config import Config
from models.GeneticRegressor import GeneticRegressor


class GeneticTree:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        working_time_matrix: np.ndarray,
        models_list: dict[int, list[GeneticRegressor]],
        verbose: bool = True,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = list()
        self.models_list = models_list
        self.verbose = verbose

    @torch.no_grad()
    def beam_search(self, beta: dict[int, dict[GeneticRegressor, int]]):
        tuple(model.eval() for models in self.models_list.values() for model in models)
        buffer = [self.root]
        for tasks in (tqdm(range(self.n_tasks - 1, 0, -1)) if self.verbose else range(self.n_tasks - 1, 0, -1)):
            temp_buffer = np.array(
                tuple(
                    [*node, task]
                    for node in buffer
                    for task in filterfalse(node.__contains__, range(self.n_tasks))
                )
            )
            del buffer
            if tasks < Config.min_size:
                break
            states = self._get_states(temp_buffer)
            headers = states[:, [-1]]
            remaining_tasks = list(
                list(filterfalse(tasks.__contains__, range(self.n_tasks)))
                for tasks in temp_buffer
            )
            states = self.working_time_matrix[remaining_tasks]
            states = np.append(headers, states, axis=1)
            del headers
            states = Tensor(states).to(
                Config.device
            )
            for model in self.models_list[tasks]:
                predictions = model(states).flatten().cpu()
                model.predictions = temp_buffer[
                    torch.argsort(Tensor(predictions))[:beta[tasks][model]]
                ]
                if len(model.predictions.shape) == 1:
                    model.predictions = model.predictions.reshape((1, -1))
            del states
            temp_buffer = np.unique(np.concatenate(tuple(model.predictions for model in self.models_list[tasks])), axis=0)
            if len(temp_buffer.shape) == 1:
                temp_buffer = temp_buffer.reshape((1, -1))
            buffer = temp_buffer
        final_permutations = np.array(
            tuple(
                chain.from_iterable(
                    map(
                        lambda remainder: np.append(state, remainder),
                        permutations(
                            filterfalse(state.__contains__, range(self.n_tasks))
                        ),
                    )
                    for state in temp_buffer
                )
            )
        )
        del temp_buffer
        final_states = self._get_states(final_permutations)
        indexes = np.argwhere(np.min(final_states[:, -1, -1]) == final_states[:, -1, -1]).flatten()
        return tuple((
            final_permutations[index],
            self._get_states([final_permutations[index]])[0],
        ) for index in indexes)


    def _get_states(self, perms: Sequence):
        states = np.zeros((len(perms), len(perms[0]) + 1, self.m_machines + 1))
        states[:, 1:, 1:] = self.working_time_matrix[perms]
        for row, column in product(
            range(1, len(perms[0]) + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += np.maximum(
                states[:, row - 1, column], states[:, row, column - 1]
            )
        return states[:, 1:, 1:]
