from pathlib import Path

import torch
from torch import nn

from models.ZeroPaddedConvRegressor import ZeroPaddedConvRegressor
from models.ZeroPaddedMultilayerPerceptron import ZeroPaddedMultilayerPerceptron
from models.ZeroPaddedPerceptron import ZeroPaddedPerceptron
from models.ZeroPaddedWideMultilayerPerceptron import ZeroPaddedWideMultilayerPerceptron


class _GeneticConfig:
    gen_train_epochs = 3
    n_population_samples = 10
    n_genetic_models = 20
    pop_retrain_rate = 0.2
    pareto_retrain_rate = 0
    size_penalty = 100
    n_pareto_samples = 10


class _ConfigWithoutModels(_GeneticConfig):
    series_model_experiment = 'series_of_models'
    recurrent_model_experiment = 'recurrent_model'
    genetic_model_experiment = 'genetic_model'
    # train = True
    train = False

    ROOT = Path(__file__).parent
    OUTPUT_GENETIC_MODELS = ROOT / "output_genetic_models"
    OUTPUT_GENETIC_MODELS.mkdir(exist_ok=True)
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    OUTPUT_RL_MODELS = ROOT / "output_rl_models"
    OUTPUT_RL_MODELS.mkdir(exist_ok=True)
    OUTPUT_RL_RESULTS = ROOT / "output_rl_results"
    OUTPUT_RL_RESULTS.mkdir(exist_ok=True)
    MODEL_TRAIN_LOG = ROOT / "model_train_log"
    MODEL_TRAIN_LOG.mkdir(exist_ok=True)
    PLOTS = ROOT / "plots"
    PLOTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")
    RL_DATA_PATH = ROOT / Path("rl_data.db")

    table_name = "Samples_{}_{}".format

    patience = 2
    n_generated_samples = 50_000
    num_processes = 4

    universal_models = tuple()
    recurrent_models = tuple()
    series_models = tuple()

    n_tasks, m_machines = 50, 10
    min_size = 4
    train_time = 12 * 3600
    # train_time = 30
    minimal_counting_time = 000
    results_average_size = 100
    train_buffer_size = 100
    beta = dict((tasks, 100) for tasks in range(n_tasks + 1))
    beta_attrition = 1.0
    gamma = 0.995
    save_interval = 10
    max_status_length = 10000

    time_constraints = [
        # 1, 2, 3, 4, 5, 25, 50, 100
    ]
    eval_iterations = 50

    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config(_ConfigWithoutModels):
    beta_constraints = range(1, 6)
    from models.RecurrentModel import RecurrentModel
    from models import ConvRegressor, MultilayerPerceptron
    from models.ConvRegressorAnySize import ConvRegressorAnySize
    from models.ConvRegressorAnySizeOneHot import ConvRegressorAnySizeOneHot
    from models.GeneticRegressorCreator import GeneticRegressor
    from models.Perceptron import Perceptron
    from models.WideMultilayerPerceptron import WideMultilayerPerceptron
    from models.EncodingPerceptron import EncodingPerceptron

    maximal_consecutive_lacks_of_improvement = 3
    hidden_size = 32
    seed = 42
    evaluation_seed = 2137
    series_models = (
        Perceptron,
        ConvRegressor,
        WideMultilayerPerceptron,
        MultilayerPerceptron,
        # GeneticRegressor,
    )
    universal_models = (
        ConvRegressorAnySize,
        ConvRegressorAnySizeOneHot,
        ZeroPaddedMultilayerPerceptron,
        ZeroPaddedWideMultilayerPerceptron,
        ZeroPaddedPerceptron,
        ZeroPaddedConvRegressor,
        # EncodingPerceptron,
    )
    recurrent_models = (
        # RecurrentModel,
    )

    genetic_models = (
        # GeneticRegressorCreator,
    )
