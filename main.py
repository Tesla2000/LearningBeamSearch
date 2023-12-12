import time
from dataclasses import dataclass, field
from typing import Self

import matplotlib.pyplot as plt
import numpy as np

from generator import RandomNumberGenerator


@dataclass
class Node:
    parent: Self
    machines: list[int]
    children: list[Self] = field(default_factory=list)
    state: np.array = None
    look_up_p: list = None

    def get_value(self) -> int:
        if self.state is not None:
            return self.state[-1][-1]
        state = self.get_state()
        if not state:
            return 0
        return state[-1][-1]

    def get_look_up_p(self) -> list:
        if self.look_up_p is not None:
            return self.look_up_p
        if not len(self.machines):
            return []
        parent_look_up_p = self.parent.get_look_up_p()
        self.look_up_p = parent_look_up_p + [p[self.machines[-1]]]
        return self.look_up_p

    def get_state(self):
        if self.state is not None:
            return self.state
        if not len(self.machines):
            return []
        parent_state = self.parent.get_state()
        self.state = parent_state + [[None for _ in range(n_tasks)]]
        self.look_up_p = self.get_look_up_p()

        def calc_previous(i, j):
            if j > 0 and i > 0:
                self.state[i][j] = max(calc_previous(i - 1, j), calc_previous(i, j - 1)) + self.look_up_p[i][j]
                return self.state[i][j]
            if i > 0:
                self.state[i][j] = calc_previous(i - 1, j) + self.look_up_p[i][j]
                return self.state[i][j]
            if j > 0:
                self.state[i][j] = calc_previous(i, j - 1) + self.look_up_p[i][j]
                return self.state[i][j]
            self.state[i][j] = self.look_up_p[i][j]
            return self.state[i][j]

        calc_previous(len(self.machines) - 1, n_tasks - 1)
        return self.state


def brute_force(m_machines):
    root = Node(None, [])
    root.value = 0
    all_parents = [root]
    all_children = []
    for task_remaining in range(m_machines):
        for parent_node in all_parents:
            for machine in filter(lambda machine: machine not in parent_node.machines, range(m_machines)):
                child_node = Node(parent_node, parent_node.machines + [machine])
                parent_node.children.append(child_node)
                all_children.append(child_node)
        all_parents = all_children
        all_children = []
    return min(all_parents, key=Node.get_value)


def branch_and_bound(m_machines):
    def cut(node_value, node, m, ub):
        """Cut is performed only when there is no chance of cutting optimum"""
        not_used_machines = filter(lambda machine: machine not in node.tasks, range(m))
        minimal_value = node_value + sum(p[m][-1] for m in not_used_machines)
        return minimal_value > ub

    return beam_search(m_machines, cut)


def beam_search(m_machines, cut_function):
    root = Node(None, [])
    root.value = 0

    def get_best(node, m, ub=float('inf')):
        node_value = node.get_value()
        if cut_function(node_value, node, m, ub):
            return None, ub
        if len(node.tasks) == m:
            return node, node_value
        best_node = None
        for machine in filter(lambda machine: machine not in node.tasks, range(m)):
            new_node, new_ub = get_best(Node(node, node.tasks + [machine]), m, ub)
            if new_ub < ub:
                ub = new_ub
                best_node = new_node
        return best_node, ub
    ub = float('inf')

    best_node, _ = get_best(root, m_machines, ub=ub)
    return best_node


# The greater skewed_factor the more speed and deviation from optimal result
def cut_early_function_1(node_value, node, m, ub, skewed_factor=.1):
    not_used_machines = filter(lambda machine: machine not in node.tasks, range(m))
    minimal_value = node_value + (1 + skewed_factor) * sum(p[m][-1] for m in not_used_machines)
    return minimal_value > ub
           # or random.random() > 2 / (len(node.machines) + 1)


def cut_early_function_2(node_value, node, m, ub, skewed_factor=.02):
    not_used_machines = tuple(filter(lambda machine: machine not in node.tasks, range(m)))
    minimal_value = node_value + sum(p[m][-1] for m in not_used_machines) + skewed_factor * sum(
        sum(p[m]) for m in not_used_machines)
    return minimal_value > ub


def cut_early_function_3(node_value, node, m, ub, skewed_factor=5):
    not_used_machines = tuple(filter(lambda machine: machine not in node.tasks, range(m)))
    minimal_value = node_value + sum(p[m][-1] for m in not_used_machines) + skewed_factor * len(not_used_machines)
    return minimal_value > ub


if __name__ == '__main__':
    # n_tasks = 5
    # m_machines = 7
    # generator = RandomNumberGenerator()
    # p = [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]
    # # Beam search test
    # start = time.time()
    # beam_value = beam_search(m_machines, cut_early_function_1).get_value()
    # print(time.time() - start)
    # start = time.time()
    # branch_value = branch_and_bound(m_machines).get_value()
    # print(time.time() - start)
    # print(beam_value / branch_value)
    #
    # #Branch and bound test
    # start = time.time()
    # machines_1 = branch_and_bound(m_machines).machines
    # print(time.time() - start)
    # start = time.time()
    # machines_2 = brute_force(m_machines).machines
    # print(time.time() - start)
    # if machines_1 != machines_2:
    #     raise ValueError

    # =============Time testing and ploting procedure for brute force===================
    # y = []
    # testing_spread = 6
    # x = [i for i in range(1, testing_spread)]
    #
    # m_machines = 7
    # generator = RandomNumberGenerator()
    #
    # for i in range(1, testing_spread): #for n_task increasing
    #     n_tasks = i
    #     p = [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]
    #
    #     time_measurements = []
    #     for k in range(3): #for calculating mean value of three probes
    #         start = time.time()
    #         brute_force(m_machines)
    #         stop = time.time()
    #
    #         time_measurements.append(stop - start)
    #
    #     y.append(sum(time_measurements) / 3)
    #
    # plt.figure(figsize = (12, 10))
    # plt.plot(x, y, marker='o')
    # plt.title(label="Brute Force ({} machines)".format(m_machines))
    # plt.xlabel("Tasks")
    # plt.ylabel("Time(s)")
    # plt.grid()
    # plt.show()

    # =============Time testing and ploting procedure for beam_search()===================
    # cut_early_functions = [cut_early_function_1, cut_early_function_2, cut_early_function_3]

    # testing_spread = 10
    # y = [[], [], []]
    # x = [i for i in range(1, testing_spread)]

    # m_machines = 7
    # generator = RandomNumberGenerator()

    # for i in range(3): #for cut_early_functions
    #     for j in range(1, testing_spread): #for n_task increasing
    #         n_tasks = j
    #         p = [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]

    #         time_measurements = []
    #         for k in range(3): #for calculating mean value of three probes
    #             start = time.time()
    #             beam_search(m_machines, cut_early_functions[i])
    #             stop = time.time()

    #             time_measurements.append(stop - start)

    #         y[i].append(sum(time_measurements) / 3)

    # plt.figure(figsize = (12, 10))
    # plt.plot(x, y[0], marker='o', label = 'cut_early_function_1')
    # plt.plot(x, y[1], marker='o', label = 'cut_early_function_2')
    # plt.plot(x, y[2], marker='o', label = 'cut_early_function_3')
    # plt.title(label="Beam search ({} machines)".format(m_machines))
    # plt.xlabel("Tasks")
    # plt.ylabel("Time(s)")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # =============Time testing and ploting procedure for branch and bound===================
    # y = []
    # testing_spread = 15
    # x = [i for i in range(1, testing_spread)]
    #
    # m_machines = 7
    # generator = RandomNumberGenerator()
    #
    # for i in range(1, testing_spread): #for n_task increasing
    #     n_tasks = i
    #     for _ in range(10):
    #         p = [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]
    #
    #         time_measurements = []
    #         # for k in range(3): #for calculating mean value of three probes
    #         #     start = time.time()
    #         b_b = branch_and_bound(m_machines).get_value()
    #         b_f = brute_force(m_machines).get_value()
    #         print("Branch and bound:", b_b)
    #         print("Brute force:", b_f)
    #         print()
    #         assert b_b == b_f
    # stop = time.time()

    # time_measurements.append(stop - start)

    #     y.append(sum(time_measurements) / 3)

    # plt.figure(figsize = (12, 10))
    # plt.plot(x, y, marker='o')
    # plt.title(label="Branch and Bound ({} machines)".format(m_machines))
    # plt.xlabel("Tasks")
    # plt.ylabel("Time(s)")
    # plt.grid()
    # plt.show()

    # =============Value testing and ploting procedure for beam_search()===================
    cut_early_functions = [cut_early_function_1, cut_early_function_2, cut_early_function_3]
    cut_early_function = cut_early_function_1

    testing_spread = 15
    b_b = []
    b_s = []
    x = [i for i in range(1, testing_spread)]

    # time_measurements = [0, 0, 0]

    n_tasks = 7
    generator = RandomNumberGenerator(100)
    skewed_factor = 2

    start = time.time()
    for j in range(1, testing_spread):  # for n_machines increasing
        m_machines = 9
        p = [[generator.nextInt(1, 99) for _ in range(n_tasks)] for _ in range(m_machines)]

        start = time.time()
        b_s.append(beam_search(m_machines, cut_early_function).get_value())
        print("Beam search:", time.time() - start)
        print(b_s)
        start = time.time()
        b_b.append(branch_and_bound(m_machines).get_value())
        print("Branch and bound:", time.time() - start)
        print(b_b)
        break
    # time_measurements[i] = stop - start

    # print('Time (cut_early_function_1): {:.2f}s.'.format(time_measurements[0]))
    # print('Time (cut_early_function_2): {:.2f}s.'.format(time_measurements[1]))
    # print('Time (cut_early_function_3): {:.2f}s.'.format(time_measurements[2]))

    plt.figure(figsize=(12, 10))
    plt.plot(x, b_s, marker='o', label='Beam search')
    plt.plot(x, b_b, marker='o', label='Branch and bound')
    plt.title(label="Values of Beam search ({} task(s), {} - skewed_factor)".format(n_tasks, skewed_factor))
    plt.xlabel("Machines")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.show()
