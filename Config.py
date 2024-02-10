from collections import defaultdict
from pathlib import Path

from regression_models.Perceptron import Perceptron


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_train_log"
    MODEL_RESULTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")

    model_type = Perceptron

    n_tasks, m_machines = 15, 25
    min_size = 5
    iterations = 2000
    minimal_counting_epoch_number = 10
    results_average_size = 500
    training_buffer_size = 5000
    beta = defaultdict(lambda: 50)
    alpha = .01
    training_buffer_retention_rate = 0.95
