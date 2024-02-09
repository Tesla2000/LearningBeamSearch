from collections import defaultdict
from pathlib import Path


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_train_log"
    MODEL_RESULTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")

    n_tasks, m_machines = 10, 25
    min_size = 5
    comparison_period = 500
    beta = defaultdict(lambda: 50)
