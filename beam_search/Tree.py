import operator
from functools import reduce
from itertools import filterfalse, permutations, product, chain
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm

from Config import Config


class Tree:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self,
            working_time_matrix: np.array,
            models: dict[int, nn.Module] = None,
            verbose: bool = True,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = list()
        self.models = models
        self.verbose = verbose
        if models is None:
            self.models = {}

    @torch.no_grad()
    def beam_search(self, beta: dict[int, float]):
        tuple(model.eval() for model in self.models.values())
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
            if len(temp_buffer) > int(beta[tasks]):
                if tasks < Config.min_size:
                    break
                states = self._get_states(temp_buffer)
                headers = states[:, [-1]]
                remaining_tasks = list(
                    list(filterfalse(tasks.__contains__, range(self.n_tasks)))
                    for tasks in temp_buffer
                )
                states = self.working_time_matrix[remaining_tasks]
                del remaining_tasks
                states = np.append(headers, states, axis=1)
                del headers
                predictions = []
                for i in range(0, len(states), Config.max_status_length):
                    state = Tensor(states[i: i + Config.max_status_length]).to(
                        self.device
                    )
                    predictions.extend(self.models[tasks](state).flatten().cpu())
                    del state
                del states
                temp_buffer = temp_buffer[
                    torch.argsort(Tensor(predictions))[
                    : max(Config.minimal_beta[tasks], int(beta[tasks]))
                    ]
                ]
                del predictions
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
        index = np.argmin(final_states[:, -1, -1])
        return (
            final_permutations[index],
            self._get_states([final_permutations[index]])[0],
        )

    @torch.no_grad()
    def beam_search_eval(self, max_beta: int):
        tuple(model.eval() for model in self.models.values())
        buffer = dict((beta, [self.root]) for beta in range(1, max_beta))
        for tasks in (tqdm(range(self.n_tasks - 1, 0, -1)) if self.verbose else range(self.n_tasks - 1, 0, -1)):
            temp_buffer = dict((beta, np.array(
                tuple(
                    [*node, task]
                    for node in b
                    for task in filterfalse(node.__contains__, range(self.n_tasks))
                )
            )) for beta, b in buffer.items())
            del buffer
            for beta, buffer in temp_buffer.items():
                if len(buffer) > beta:
                    if tasks < Config.min_size:
                        break
                    states = self._get_states(buffer)
                    headers = states[:, [-1]]
                    remaining_tasks = list(
                        list(filterfalse(tasks.__contains__, range(self.n_tasks)))
                        for tasks in buffer
                    )
                    states = self.working_time_matrix[remaining_tasks]
                    states = np.append(headers, states, axis=1)
                    del headers
                    predictions = []
                    for i in range(0, len(states), Config.max_status_length):
                        state = Tensor(states[i: i + Config.max_status_length]).to(
                            self.device
                        )
                        predictions.extend(self.models[tasks](state).flatten().cpu())
                        del state
                    del states
                    buffer = buffer[
                        torch.argsort(Tensor(predictions))[:beta]
                    ]
                    if len(buffer.shape) == 1:
                        buffer = np.array([buffer])
                    temp_buffer[beta] = buffer
                    del predictions
            buffer = temp_buffer
        final_permutations = dict((beta, np.array(
            tuple(
                chain.from_iterable(
                    map(
                        lambda remainder: np.append(state, remainder),
                        permutations(
                            filterfalse(state.__contains__, range(self.n_tasks))
                        ),
                    )
                    for state in buffer
                )
            )
        )) for beta, buffer in temp_buffer.items())
        del temp_buffer
        final_states = dict((beta, self._get_states(final_permutation)) for beta, final_permutation in final_permutations.items())
        indexes = dict((beta, np.argmin(final_state[:, -1, -1])) for beta, final_state in final_states.items())
        return dict((beta, (
            final_permutations[beta][index],
            self._get_states([final_permutations[beta][index]])[0][-1, -1],
        )) for beta, index in indexes.items())


    def fast_brute_force(self):
        perms = list(
            list(permutation) for permutation in permutations(range(self.n_tasks))
        )
        states = self._get_states(perms)
        index = np.argmin(states[:, -1, -1])
        return perms[index], self._get_states([perms[index]])[0]

    def _get_states(self, perms: np.array):
        states = np.zeros((len(perms), len(perms[0]) + 1, self.m_machines + 1))
        states[:, 1:, 1:] = self.working_time_matrix[perms]
        for row, column in product(
                range(1, len(perms[0]) + 1), range(1, self.m_machines + 1)
        ):
            states[:, row, column] += np.maximum(
                states[:, row - 1, column], states[:, row, column - 1]
            )
        return states[:, 1:, 1:]
