import numpy as np
from matplotlib import pyplot as plt

from Config import Config

if __name__ == '__main__':
    labels = []
    for log_file in Config.MODEL_RESULTS.iterdir():
        data = np.loadtxt(log_file, delimiter=',')
        x = np.linspace(*data[[0, -1], 0], num=len(data))[
                                                          Config.minimal_counting_epoch_number:
            ]
        y = data[
                 Config.minimal_counting_epoch_number:
        , 1]
        del data
        plt.plot(x, y)
        labels.append(log_file.name.partition('_')[0])
    plt.legend(labels)
    plt.show()
