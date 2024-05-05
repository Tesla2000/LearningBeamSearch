from collections import defaultdict

from Config import Config
from beam_search.Tree import Tree
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from models.Perceptron import Perceptron

if __name__ == "__main__":
    generator = RandomNumberGenerator(Config.evaluation_seed)
    models = dict(
        (tasks, Perceptron(tasks, 10).to(Config.device))
        for tasks in range(Config.min_size, Config.n_tasks + 1)
    )
    working_time_matrix = generate_taillard(generator)
    tree = Tree(working_time_matrix, models)
    _, state = tree.beam_search(defaultdict(lambda: 2))
