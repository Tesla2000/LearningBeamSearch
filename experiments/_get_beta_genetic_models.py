from collections import Counter
from itertools import islice, cycle

from models.GeneticRegressor import GeneticRegressor


def _get_beta_genetic_models(models_lists: dict[int, dict[GeneticRegressor, float]], beta: int) -> tuple[
    dict[int, Counter], dict[int, list[GeneticRegressor]]]:
    beta_dict = {}
    filtered_models = {}
    for tasks, regressors in models_lists.items():
        best_models = dict(filter(lambda item: item[1], regressors.items()))
        sorted_models = list(
            model for model, _ in sorted(best_models.items(), key=lambda item: item[1], reverse=True))
        beta_dict[tasks] = Counter(islice(cycle(sorted_models), beta))
        filtered_models[tasks] = list(beta_dict[tasks].keys())
    return beta_dict, filtered_models
