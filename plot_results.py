import re
from statistics import fmean

import numpy as np
from matplotlib import pyplot as plt

from Config import Config
from table2latex import table2latex


def plot(labels_translator: dict, name: str, norm: bool = False):
    for n_tasks in (50, 20,):
        normalizer = np.array([tuple(map(lambda line: float(line.split(',')[1]), next(
            Config.MODEL_TRAIN_LOG.glob(f"ZeroPaddedPerceptron_{n_tasks}*")).read_text().splitlines()))[-1]])
        if norm:
            normalizer = np.array(tuple(map(lambda line: float(line.split(',')[1]),
                                            next(Config.MODEL_TRAIN_LOG.glob(
                                                f"ZeroPaddedConvRegressor_{n_tasks}*")).read_text().splitlines())))
        for log_file in Config.MODEL_TRAIN_LOG.glob(f"*{n_tasks}*"):

            data = np.loadtxt(log_file, delimiter=",")
            x = np.linspace(*data[[0, -1], 0], num=len(data))
            y = data[:, 1][np.where(x > Config.minimal_counting_time)] / normalizer[np.array(
                np.round(np.linspace(0, len(normalizer), len(np.where(x > Config.minimal_counting_time)[0]))).clip(0,
                                                                                                                   len(normalizer) - 1),
                dtype=int)]
            x = x[np.where(x > Config.minimal_counting_time)]
            del data
            conv_name = log_file.name.partition('_')[0]
            if conv_name in labels_translator:
                plt.plot(x, y, label=labels_translator.get(conv_name, conv_name))
        if not norm:
            plt.axhline(y=1, color='black', linestyle='dotted')
        plt.ylabel(f"Uśrednione {Config.results_average_size} wyników\nw relacji do algorytmu zachłannego")
        plt.xlabel("Czas szkolenia [s]")
        plt.legend(bbox_to_anchor=(0, -.2), loc='upper left')
        plt.subplots_adjust(bottom=.5)
        plt.title(f"{name} {n_tasks} zadań")
        plt.savefig(Config.PLOTS / f'training_{n_tasks}_{name}.png')
        plt.show()
        plt.clf()

        print(table2latex((["", ["beta", "", "", "", ""]], [""] + list(map(str, Config.beta_constraints)), *tuple(
            (labels_translator[model_type], *tuple(
                round(fmean(
                    eval(
                        Config.OUTPUT_RL_RESULTS.joinpath(
                            f"{model_type}_{constraint}_20" if n_tasks == 20 else f"{model_type}_{constraint}").read_text()
                    )
                ) / fmean(
                    eval(
                        Config.OUTPUT_RL_RESULTS.joinpath(
                            f"ZeroPaddedPerceptron_{constraint}_20" if n_tasks == 20 else f"{model_type}_{constraint}").read_text()
                    )
                ), 3) for constraint in Config.beta_constraints
            )) for model_type in labels_translator
        )),
                          caption=f"Wyniki sieci neuronowych w zależności od szerokości snopu w relacji do algorytmu chciwego dla {n_tasks} zadań",
                          label=f"evaluation_beta_{n_tasks}_{name}", placement="h"))

        print(table2latex((["", ["czas wykonania [s]", "", ] + ([""] if n_tasks == 50 else [])],
                           [""] + list(map(str, Config.time_constraints if n_tasks == 50 else [5, 10])), *tuple(
            (labels_translator[model_type], *tuple(
                round(fmean(
                    eval(
                        path.read_text()
                    )
                ) / fmean(
                    eval(
                        reference_path.read_text()
                    )
                ), 3) for path, reference_path in zip(sorted(
                    filter(lambda path: path.name.endswith("_20") if n_tasks == 20 else not path.name.endswith("_20"),
                           Config.OUTPUT_RL_RESULTS.glob(f"{model_type}*")),
                    key=lambda path: int(re.findall(r'\d+', path.name)[0]))[-(3 if n_tasks == 50 else 2):], sorted(
                    filter(lambda path: path.name.endswith("_20") if n_tasks == 20 else not path.name.endswith("_20"),
                           Config.OUTPUT_RL_RESULTS.glob("ZeroPaddedPerceptron*")),
                    key=lambda path: int(re.findall(r'\d+', path.name)[0]))[-(3 if n_tasks == 50 else 2):])
            )) for model_type in labels_translator
        )),
                          caption=f"Wyniki sieci neuronowych w zależności od czasu w relacji do algorytmu chciwego dla {n_tasks} zadań",
                          label=f"evaluation_time_{n_tasks}_{name}", placement="h"))


if __name__ == "__main__":
    labels_translator_model_comparison = {
        "MultilayerPerceptron": "Trójwarstwowy Perceptron",
        "ConvRegressor": "CNN",
        "Perceptron": "Perceptron",
        "WideMultilayerPerceptron": "Perceptron z blokiem gęstym",
        "RecurrentModel": "RNN",
        "GeneticRegressor": "Mieszanka modeli",
    }
    plot(labels_translator_model_comparison, "Prównanie modeli")
    labels_translator_different_size_comparison = {
        "ConvRegressor": "CNN",
        "ConvRegressorAnySizeOneHot": "CNN enkodowany jedynkowo",
        "ConvRegressorAnySize": "CNN z liczbą zadań jako skalarem",
        "EncodingConvRegressor": "CNN z enkodowaniem",
        "ZeroPaddedConvRegressor": "CNN z wypełnianiem zerami",
    }
    plot(labels_translator_different_size_comparison, "Porównanie sposóbów rozwiązania różnych rozmiarów wejść",
         norm=True)
