from pathlib import Path


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_results"
    MODEL_RESULTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")

    n_tasks, m_machines = 10, 25
    min_size = 5
    limit = 10_000
