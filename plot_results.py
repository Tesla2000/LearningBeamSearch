import re
import sys
from contextlib import suppress
from itertools import repeat, islice
from pathlib import Path
from statistics import fmean
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt

from Config import Config


def plot(labels_translator: dict, name: str, ylabel: str, norm: bool = False,
         maximal_counting_times: Iterable = repeat(sys.maxsize)):
    for n_tasks, maximal_counting_time in zip((50, 20,), maximal_counting_times):
        normalizer = np.array([tuple(map(lambda line: float(line.split(',')[1]), next(
            Config.MODEL_TRAIN_LOG.glob(f"ZeroPaddedPerceptron_{n_tasks}*")).read_text().splitlines()))[-1]])
        if norm:
            normalizer = np.array(tuple(map(lambda line: float(line.split(',')[1]),
                                            next(Config.MODEL_TRAIN_LOG.glob(
                                                f"ZeroPaddedConvRegressor_{n_tasks}*")).read_text().splitlines())))
        for log_file in Config.MODEL_TRAIN_LOG.glob(f"*{n_tasks}*"):

            data = np.loadtxt(log_file, delimiter=",")
            x = np.linspace(*data[[0, -1], 0], num=len(data))
            y = (data[:, 1] / normalizer[np.array(
                np.round(np.linspace(0, len(normalizer), len(x))).clip(0, len(normalizer) - 1), dtype=int)])[
                np.where((x > Config.minimal_counting_time) & (x < maximal_counting_time))]
            x = x[np.where((x > Config.minimal_counting_time) & (x < maximal_counting_time))]
            del data
            conv_name = log_file.name.partition('_')[0]
            if conv_name in labels_translator:
                plt.plot(x, y, label=labels_translator.get(conv_name, conv_name))
        if not norm:
            plt.axhline(y=1, color='black', linestyle='dotted')
        plt.ylim(.9, 1.3)
        plt.ylabel(ylabel)
        plt.xlabel("Czas szkolenia [s]")
        plt.legend()
        # plt.subplots_adjust(bottom=.35)
        plt.savefig(Config.PLOTS / f'training_{n_tasks}_{name}.png')
        plt.show()
        plt.clf()


def plt_results_over_n_tasks():
    reference_model = "ZeroPaddedConvRegressor"
    compared_model = "ConvRegressor"
    for n_tasks_folder in sorted(Config.OUTPUT_RL_RESULTS.iterdir(), key=lambda path: int(path.name)):
        results_over_time = []
        for time in islice(sorted((n_tasks_folder / str(Config.m_machines) / "time").iterdir(), key=lambda path: int(path.name)), 0, 5):
            results_over_time.append(fmean(eval((time / compared_model).read_text())) / fmean(eval((time / reference_model).read_text())))
        plt.plot([2, 5, 10, 25, 50], results_over_time)
    plt.xticks([2, 5, 10, 25, 50])
    plt.xlabel("Czas [s]")
    plt.ylabel("Uśredniony znormalizowany C_max")
    plt.legend([path.name for path in sorted(Config.OUTPUT_RL_RESULTS.iterdir(), key=lambda path: int(path.name))])
    plt.savefig(Config.PLOTS / "evaluation_over_time.png")
    plt.show()


def plot_results(labels_translator: dict, n_tasks: int, normalize: str, name: str):
    for kind in ("beta", "time"):
        beta_results = np.array(tuple(
            tuple(fmean(eval(path.joinpath(model_name).read_text())) for model_name in labels_translator) for path in
            Path(f"output_rl_results/{n_tasks}/10/{kind}").iterdir()))
        beta_results /= np.array(
            tuple([fmean(eval(path.joinpath(normalize).read_text()))] for path in
                  Path(f"output_rl_results/{n_tasks}/10/{kind}").iterdir()))
        ticks = list(sorted(int(path.name) for path in Path(f"output_rl_results/{n_tasks}/10/{kind}").iterdir()))
        plt.xticks(ticks)
        plt.xlabel("beta [-]" if kind == "beta" else "time [s]")
        plt.ylabel("Uśredniony znormalizowany C_max")
        plt.plot(ticks, beta_results)
        plt.legend(labels_translator.values())
        # plt.savefig(Config.PLOTS / f"evaluation_{kind}_{name}.png")
        plt.show()
    # print(table2latex((["Architektura sieci", [r"$\beta$ [-]", "", "", "", ""]],
    #                    [""] + list(map(str, Config.beta_constraints)), *tuple(
    #     (labels_translator[model_type], *tuple(
    #         round(fmean(
    #             eval(
    #                 Config.OUTPUT_RL_RESULTS.joinpath(
    #                     f"{model_type}_{constraint}_{n_tasks}").read_text()
    #             )
    #         ) / fmean(
    #             eval(
    #                 Config.OUTPUT_RL_RESULTS.joinpath(
    #                     f"ZeroPaddedPerceptron_{constraint}_{n_tasks}").read_text()
    #             )
    #         ), 3) for constraint in Config.beta_constraints
    #     )) for model_type in labels_translator
    # )),
    #                   caption=f"Wyniki sieci neuronowych w zależności od szerokości snopu w relacji do wyników perceptronu z wypełnianiem zerami dla {n_tasks} zadań",
    #                   label=f"table:evaluation_beta_{n_tasks}_{name}", placement="h"))
    #
    # print(table2latex((["Architektura sieci", ["czas wykonania [s]", "", ] + ([""] if n_tasks == 50 else [])],
    #                    [""] + list(map(str, Config.time_constraints if n_tasks == 50 else [5, 10])), *tuple(
    #     (labels_translator[model_type], *tuple(
    #         round(fmean(
    #             eval(
    #                 path.read_text()
    #             )
    #         ) / fmean(
    #             eval(
    #                 reference_path.read_text()
    #             )
    #         ), 3) for path, reference_path in zip(sorted(
    #             filter(lambda path: path.name.endswith("_20") if n_tasks == 20 else not path.name.endswith("_20"),
    #                    Config.OUTPUT_RL_RESULTS.glob(f"{model_type}*")),
    #             key=lambda path: int(re.findall(r'\d+', path.name)[0]))[-(3 if n_tasks == 50 else 2):], sorted(
    #             filter(lambda path: path.name.endswith("_20") if n_tasks == 20 else not path.name.endswith("_20"),
    #                    Config.OUTPUT_RL_RESULTS.glob("ZeroPaddedPerceptron*")),
    #             key=lambda path: int(re.findall(r'\d+', path.name)[0]))[-(3 if n_tasks == 50 else 2):])
    #     )) for model_type in labels_translator
    # )),
    #                   caption=f"Wyniki sieci neuronowych w zależności od czasu w relacji do wyników perceptronu z wypełnianiem zerami dla {n_tasks} zadań",
    #                   label=f"table:evaluation_time_{n_tasks}_{name}", placement="h"))


def plot_perceptron_frequency():
    paths = len(range(50, 510, 10)) * [0]
    model_sizes = []
    for i, n_tasks in enumerate(range(50, 510, 10)):
        with suppress(StopIteration):
            paths[i] = int(re.findall(r'\d+', next(
                Config.OUTPUT_GENETIC_MODELS.joinpath("6").glob(f"GeneticRegressor{n_tasks}_1_0.*")).name)[-1])
        model_sizes.append(None)
    plt.plot(range(Config.min_size, len(paths) + Config.min_size), paths)
    plt.xlabel("Liczba zadań do uszeregowania [-]")
    plt.ylabel("Część ścieżek [%]")
    # plt.savefig(Config.PLOTS / 'perceptron_frequency.png')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    # plot_perceptron_frequency()
    y_label = "Uśredniony znormalizowany C_max"
    labels_translator_model_comparison = {
        "MultilayerPerceptron": "Trójwarstwowy Perceptron",
        "ConvRegressor": "CNN",
        "Perceptron": "Perceptron",
        "WideMultilayerPerceptron": "Perceptron z blokiem gęstym",
        "RecurrentModel": "RNN",
        "GeneticRegressor": "Mieszanka modeli",
    }
    plot(labels_translator_model_comparison, "Prównanie modeli",
         y_label,
         maximal_counting_times=(21600, 3600))
    # labels_translator_different_size_comparison = {
    #     "ConvRegressor": "MARL CNN",
    #     "ConvRegressorAnySizeOneHot": "CNN enkodowany jedynkowo",
    #     "ConvRegressorAnySize": "CNN z liczbą zadań jako skalarem",
    #     "EncodingConvRegressor": "CNN z enkodowaniem",
    #     "ZeroPaddedConvRegressor": "CNN z wypełnianiem zerami",
    # }
    # plot(labels_translator_different_size_comparison, "Porównanie sposóbów rozwiązania różnych rozmiarów wejść",
    #      y_label,
    #      norm=True, maximal_counting_times=(21600, 3600))
    # plt_results_over_n_tasks()
    # labels_translator_different_size_comparison = {
    #     "MultilayerPerceptron": "Trójwarstwowy Perceptron",
    #     "ConvRegressor": "CNN",
    #     "Perceptron": "Perceptron",
    #     "WideMultilayerPerceptron": "Perceptron z blokiem gęstym",
    #     "RecurrentModel": "RNN",
    #     "GeneticRegressor": "Mieszanka modeli",
    # }
    # plot_results(labels_translator_different_size_comparison, 50, "ZeroPaddedPerceptron", "modele")
    # labels_translator_different_size_comparison = {
    #     "ConvRegressor": "MARL CNN",
    #     "ConvRegressorAnySizeOneHot": "CNN enkodowany jedynkowo",
    #     "ConvRegressorAnySize": "CNN z liczbą zadań jako skalarem",
    #     "EncodingConvRegressor": "CNN z enkodowaniem",
    #     "ZeroPaddedConvRegressor": "CNN z wypełnianiem zerami",
    # }
    # plot_results(labels_translator_different_size_comparison, 50, "ZeroPaddedConvRegressor", "size")
