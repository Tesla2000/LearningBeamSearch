from Config import Config


def save_results(model_type, number: int, beta: bool, results):
    folder = Config.OUTPUT_RL_RESULTS.joinpath(str(Config.n_tasks)).joinpath(str(Config.m_machines)).joinpath("beta" if beta else "time").joinpath(str(number))
    folder.mkdir(exist_ok=True, parents=True)
    folder.joinpath(model_type.__name__).write_text(str(results))
