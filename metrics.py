import numpy as np


def RMSE(y_1, y_2):
    return np.sqrt(np.mean((y_1 - y_2) ** 2))
