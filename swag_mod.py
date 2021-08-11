import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from torch import nn

from data import get_loaders
from training import test_step, train_step
from utils import print_losses

def calculate_coeffs(weight_series, verbose=False):
    regression_start = weight_series.shape[0] // 2
    coeffs = np.zeros(weight_series.shape[1])
    for i in range(weight_series.shape[1]):
        y = weight_series[regression_start:, i]
        x = np.arange(len(y)).reshape(-1, 1)
        regression = LinearRegression()
        regression.fit(x, y)
        coeffs[i] = regression.coef_[0]
    if verbose:
        plt.hist(coeffs, bins=100)
        plt.show()
    return coeffs
