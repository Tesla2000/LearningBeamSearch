import re

import numpy as np
from matplotlib import pyplot as plt

from Config import Config

if __name__ == "__main__":
    labels = []
    for log_file in Config.MODEL_RESULTS.iterdir():
        data = np.loadtxt(log_file, delimiter=",")
        x = np.linspace(*data[[0, -1], 0], num=len(data))
        y = data[:, 1][np.where(x > Config.minimal_counting_time)]
        x = x[np.where(x > Config.minimal_counting_time)]
        del data
        plt.plot(x, y)
        labels.append(''.join(re.findall(r'\D+', log_file.name)).replace('_', ' ').strip())
    plt.ylabel(f"Uśredniony wyniki {Config.results_average_size} ostatnich problemów")
    plt.xlabel("Czas szkolenia [s]")
    plt.legend(labels)
    plt.show()
