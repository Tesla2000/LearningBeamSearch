from itertools import filterfalse, permutations, product, chain

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from Config import Config
from regression_models.RecurrentModel import RecurrentModel


class RecurrentTree:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self,
            working_time_matrix: np.array,
            verbose: bool = True,
    ) -> None:
        """The greater beta is the more results are accepted which leads to better results and longer calculations"""
        self.n_tasks, self.m_machines = working_time_matrix.shape
        self.working_time_matrix = working_time_matrix
        self.root = list()
        self.verbose = verbose

    @torch.no_grad()
    def beam_search(self, model: RecurrentModel, beta: dict[int, float]):
        buffer = [self.root]
        hns = [Tensor(self.working_time_matrix).flatten().unsqueeze(0).to(Config.device)]
        for tasks in (tqdm(range(self.n_tasks - 1, 0, -1)) if self.verbose else range(self.n_tasks - 1, 0, -1)):
            temp_buffer = np.array(
                tuple(
                    [*node, task]
                    for node in buffer
                    for task in filterfalse(node.__contains__, range(self.n_tasks))
                )
            )
            if len(temp_buffer) > max(Config.minimal_beta[tasks], int(beta[tasks])):
                if tasks < Config.min_size:
                    break
                results = tuple(chain.from_iterable(model(Tensor(self.working_time_matrix[task]).unsqueeze(0).to(Config.device), hn) for node, hn in zip(buffer, hns)
                                         for task in filterfalse(node.__contains__, range(self.n_tasks))))
                predictions = results[::2]
                indexes_to_leave = torch.argsort(Tensor(predictions))[
                                   : max(Config.minimal_beta[tasks], int(beta[tasks]))
                                   ]
                del predictions
                hns = results[1::2]
                temp_buffer = temp_buffer[indexes_to_leave]
                hns = tuple(hns[index] for index in indexes_to_leave)
                del results
            else:
                hns = tuple(model.update_hn(Tensor(self.working_time_matrix[task]).unsqueeze(0).to(Config.device), hn) for node, hn in zip(buffer, hns)
                            for task in filterfalse(node.__contains__, range(self.n_tasks)))
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
        return final_permutations[index], self._get_states([final_permutations[index]])[0]

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
