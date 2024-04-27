from statistics import fmean

import numpy as np
from matplotlib import pyplot as plt

from Config import Config
from table2latex import table2latex

if __name__ == "__main__":
    labels_translator = {
        "MultilayerPerceptron": "Wielowarstwowy Perceptron",
        "ConvRegressor": "CNN",
        "ConvRegressorAnySizeOneHot": "CNN enkodowany jedynkowo",
        "ConvRegressorAnySize": "CNN z globalnym poolingiem",
        "Perceptron": "Perceptron",
        "WideMultilayerPerceptron": "Perceptron z blokiem gęstym",
        # "RecurrentModel": "RNN",
        "ZeroPaddedPerceptron": "Perceptron z wypełniamiem zerami",
        # "EncodingPerceptron": "EP",
        # "MultilayerPerceptron": "trójwarstwory perceptron",
        # "ConvRegressor": "model konwolucyjny",
        # "ConvRegressorAnySizeOneHot": "model konwolucyjny stosowany na każdy etapie drzewa\nz enkodowaniem jedynkowym",
        # "ConvRegressorAnySize": "model konwolucyjny stosowany na każdy etapie drzewa",
        # "Perceptron": "perceptron",
        # "WideMultilayerPerceptron": "perceptron z gęstym blokiem",
        # "RecurrentModel": "model rekurencyjny",
    }

    # for log_file in Config.MODEL_TRAIN_LOG.iterdir():
    #     data = np.loadtxt(log_file, delimiter=",")
    #     x = np.linspace(*data[[0, -1], 0], num=len(data))
    #     y = data[:, 1][np.where(x > Config.minimal_counting_time)]
    #     x = x[np.where(x > Config.minimal_counting_time)]
    #     del data
    #     plt.plot(x, y, label=labels_translator.get(log_file.name.partition('_')[0], log_file.name.partition('_')[0]))
    # plt.ylabel(f"Uśrednione {Config.results_average_size} ostatnich wyników")
    # plt.xlabel("Czas szkolenia [s]")
    # plt.legend(bbox_to_anchor=(0, -.2), loc='upper left')
    # plt.subplots_adjust(bottom=.5)
    # plt.savefig(Config.PLOTS / 'training.png')
    # plt.show()
    # plt.clf()

    print(table2latex((["", ["beta", "", "", "", ""], ["czas [s]", "", ""]], [""] + Config.time_constraints, *tuple(
        (labels_translator[model_type], *tuple(
            round(fmean(
                eval(
                    Config.OUTPUT_RL_RESULTS.joinpath(f"{model_type}_{constraint}").read_text()
                )
            )) for constraint in Config.time_constraints
        )) for model_type in labels_translator
    )), caption="Wyniki sieci neuronowych w zależności od szerokości snopu i czasu", label="evaluation", placement="h"))
