from statistics import fmean

import numpy as np
from matplotlib import pyplot as plt

from Config import Config

if __name__ == "__main__":
    labels_translator = {
        "MultilayerPerceptron": "dwuwarstwory perceptron",
        "ConvRegressor": "model konwolucyjny",
        "ConvRegressorAnySizeOneHot": "model konwolucyjny stosowany na każdy etapie drzewa\nz enkodowaniem jedynkowym",
        "ConvRegressorAnySize": "model konwolucyjny stosowany na każdy etapie drzewa",
        "Perceptron": "perceptron",
        "WideMultilayerPerceptron": "perceptron z gęstym blokiem",
        "RecurrentModel": "model rekurencyjny",
        # "ZeroPaddedPerceptron": "Perceptron wypełniany zerami",
        # "EncodingPerceptron": "Perceptron na enkodowanych danych",
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

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
    for color_index, model_type in enumerate(labels_translator.keys()):
        x_values = Config.time_constraint
        y_values = tuple(fmean(eval(Config.OUTPUT_RL_RESULTS.joinpath(f"{model_type}_{constraint}").read_text())) for constraint in Config.time_constraint)
        plt.plot(
            x_values,
            y_values,
            label=labels_translator[model_type],
            color=colors[color_index],
        )
    plt.ylabel("Średnia wyników")
    plt.xlabel("Czas obliczeń [s]")
    plt.legend(bbox_to_anchor=(0, -0.2), loc="upper left")
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(Config.PLOTS / "evaluation.png")
    plt.show()
    plt.clf()
