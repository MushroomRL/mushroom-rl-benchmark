import numpy as np


def max_metric(data):
    return np.max(data, axis=1)


def convergence_metric(data):
    data_max_idx = np.argmax(data, axis=1)

    variance_max_list = list()
    for i, idx in enumerate(data_max_idx):
        variance_max = np.std(data[i][idx:])
        variance_max_list.append(variance_max)

    return np.array(variance_max_list)

