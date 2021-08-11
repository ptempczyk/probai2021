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


def train_SWAG_mod(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    model: nn.Module,
    K,
    epochs: int = 100,
    batch_size: int = 100,
    lr: float = 0.1,
    verbose: bool = True,
    c=1,
    momentum=0,
    weight_decay=0,
):
    assert c >= 1 and K >= 2
    train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size)
    theta_epoch = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()
    theta = theta_epoch.clone()
    theta_square = theta_epoch.clone() ** 2
    D = None

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in train_loader]
    test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
    if verbose:
        print_losses(0, train_losses, test_losses)

    thetas = []
    for epoch in range(1, epochs + 1):
        train_losses = [train_step(batch_x, batch_y, model, optimizer, criterion) for batch_x, batch_y in train_loader]
        test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
        if verbose:
            print_losses(epoch, train_losses, test_losses)
        if epoch % c == 0:
            if verbose:
                print("SWAG moment update")
            n = epoch / c
            theta_epoch = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()
            thetas.append(theta_epoch.clone())
            theta = (n * theta + theta_epoch.clone()) / (n + 1)
            theta_square = (n * theta_square + theta_epoch.clone() ** 2) / (n + 1)
            deviations = (theta_epoch.clone() - theta).reshape(-1, 1)
            if D is None:
                D = deviations
            else:
                if D.shape[1] == K:
                    D = D[:, 1:]
                D = torch.cat((D, deviations), dim=1)
    sigma_diag = theta_square - theta ** 2
    torch.nn.utils.vector_to_parameters(theta, model.parameters())
    test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
    # print(f"Finished SWAG.     Best test loss: {np.mean(test_losses):.5f}")
    return theta, sigma_diag, D, thetas
