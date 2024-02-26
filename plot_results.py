import re
from statistics import fmean

import numpy as np
from matplotlib import pyplot as plt
from numpy import std

from Config import Config

if __name__ == "__main__":
    labels_translator = {
        "MultilayerPerceptron": "dwuwarstwory perceptron",
        "ConvRegressor": "model konwolucyjny",
        "Perceptron": "perceptron",
        "UniversalEfficientNetAnySize": "efficientnet trenowany na wszystkich rozmiarach",
        "UniversalEfficientNetMaxSize": "dwuwarstwory perceptron",
        "WideMultilayerPerceptron": "perceptron z gęstym blokiem",
    }
    # for log_file in Config.MODEL_RESULTS.iterdir():
    #     data = np.loadtxt(log_file, delimiter=",")
    #     x = np.linspace(*data[[0, -1], 0], num=len(data))
    #     y = data[:, 1][np.where(x > Config.minimal_counting_time)]
    #     x = x[np.where(x > Config.minimal_counting_time)]
    #     del data
    #     plt.plot(x, y, label=labels_translator[log_file.name.partition('_')[0]])
    # plt.ylabel(f"Uśrednione {Config.results_average_size} ostatnich wyników")
    # plt.xlabel("Czas szkolenia [s]")
    # plt.legend(bbox_to_anchor=(0, -.2), loc='upper left')
    # plt.subplots_adjust(bottom=.5)
    # plt.savefig(Config.PLOTS / 'training.png')
    # plt.clf()
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    for color_index, results_file in enumerate(Config.OUTPUT_RL_RESULTS.iterdir()):
        results = eval(results_file.read_text())
        mean = fmean(results)
        plt.hist(results, label=labels_translator[results_file.name], color=colors[color_index])
        plt.axvline(x=mean, linestyle='-', color=colors[color_index])
    plt.ylabel("Liczba wyników")
    plt.xlabel("Wyniki")
    plt.legend(bbox_to_anchor=(0, -.2), loc='upper left')
    plt.subplots_adjust(bottom=.5)
    plt.savefig(Config.PLOTS / 'evaluation.png')
    plt.clf()
