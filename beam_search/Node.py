from dataclasses import dataclass, field
from itertools import islice, accumulate
from typing import Optional, Self

import numpy as np


@dataclass
class Node:
    parent: Optional[Self]
    tasks: tuple[int, ...]
    m_machines: int
    working_time_matrix: np.array
    children: list[Self] = field(default_factory=list)
    state: np.array = None
    _value: float = 0.0

    @property
    def value(self) -> float:
        if self._value:
            return self._value
        state = self.get_state()
        if state.shape == (1, 0):
            return 0
        self._value = state[-1, -1]
        return self.value

    def fill_state(self) -> None:
        last_task = self.tasks[-1]

        def process(total_time_at_previous_machine, zipped_expression):
            work_time, total_time_at_previous_task = zipped_expression
            return work_time + max(
                total_time_at_previous_machine, total_time_at_previous_task
            )

        self.state[-1] = np.array(
            tuple(
                islice(
                    accumulate(
                        zip(self.working_time_matrix[last_task], self.state[-2]),
                        process,
                        initial=0,
                    ),
                    1,
                    None,
                )
            )
        )

    def get_state(self) -> np.array:
        if self.state is not None:
            return self.state
        if not len(self.tasks):
            return np.array([[]])
        parent_state = self.parent.get_state()
        if parent_state.shape != (1, 0):
            self.state = np.append(parent_state, np.empty((1, self.m_machines)), axis=0)
            self.fill_state()
        else:
            self.state = self.working_time_matrix[self.tasks[-1]].reshape(1, -1)
        return self.state
