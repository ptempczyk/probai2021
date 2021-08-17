import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from torch import nn

from data import get_loaders, train_test_split, load_dataset
from definitions import device, TOLERANCE
from metrics import RMSE
from models import load_model, create_model
from swag import sample_from_SWAG
from training import test_step, get_test_predictions, train_model
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


def train_step_mod(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    lr_multipliers,
):
    optimizer.zero_grad()
    outputs = model(batch_x.to(device))
    loss = criterion(outputs, batch_y.to(device))
    loss.backward()
    multitiply_grads(model, lr_multipliers)
    optimizer.step()
    return loss.item()


def train_SWAG_mod(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    model: nn.Module,
    K,
    lr_multipliers,
    epochs: int = 100,
    batch_size: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    c: int = 1,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
):
    assert c >= 1 and K >= 2
    #     train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size)
    train_loader, _, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size, val_loader=True)
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
        train_losses = [
            train_step_mod(batch_x, batch_y, model, optimizer, criterion, lr_multipliers)
            for batch_x, batch_y in train_loader
        ]
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
    model.to(device)
    # test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
    # print(f"Finished SWAG.     Best test loss: {np.mean(test_losses):.5f}")
    return theta, sigma_diag, D, thetas


def multitiply_grads(model, lr_multipliers):
    start_ind = 0
    for params in model.parameters():
        shape = params.shape
        total_len = params.reshape(-1).shape[0]
        multipliers = lr_multipliers[start_ind : (start_ind + total_len)].reshape(shape)
        start_ind += total_len
        params.grad = params.grad * multipliers.to(device)


def sample_and_get_metrics(x_train, y_train, x_test, y_test, model, theta_swa, sigma_diag, D, K, S, y_scaler):
    samples = sample_from_SWAG(x_train, y_train, x_test, y_test, model, theta_swa, sigma_diag, D, K, S)
    samples_array = np.concatenate(samples, axis=1)
    y_pred = samples_array.mean(axis=1, keepdims=True)
    y_l = np.percentile(samples_array, 2.5, axis=1, keepdims=True)
    y_u = np.percentile(samples_array, 97.5, axis=1, keepdims=True)
    rmse = RMSE(y_pred, y_test, y_scaler)
    pcip = np.mean((y_l < y_test) & (y_test < y_u))
    y_l = y_scaler.inverse_transform(y_l)
    y_u = y_scaler.inverse_transform(y_u)
    mpiw = np.mean(y_u - y_l)
    return rmse, pcip, mpiw


def run_MCSWAG(dataset_name, weight_decay=0.0, lr=0.01, K=10, S=500, multiplier=2, verbose=False):
    tolerance = TOLERANCE[dataset_name]
    x_train, y_train, x_test, y_test, x_scaler, y_scaler = load_dataset(dataset_name, verbose=False)
    batch_size = x_train.shape[0] // 9
    model = create_model(x_train, layer_dims=[50], verbose=False)
    train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        dataset_name,
        lr=0.001,
        epochs=50000,
        verbose=False,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    model = load_model(model, f"best_model_weights-{dataset_name}.pth", verbose=False)
    theta_epoch = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()
    lr_multipliers = torch.ones_like(theta_epoch).float()
    round_results = []
    for rounds in range(10):
        try:
            model = load_model(model, f"best_model_weights-{dataset_name}.pth", verbose=False)
            theta_swa, sigma_diag, D, thetas = train_SWAG_mod(
                x_train,
                y_train,
                x_test,
                y_test,
                model,
                K,
                lr_multipliers,
                verbose=False,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                epochs=50,
            )
            weight_series = torch.stack(thetas).numpy()
            weight_series -= weight_series.mean(axis=0, keepdims=True)
            weight_series /= weight_series.std(axis=0, keepdims=True) + 1e-10
            if verbose:
                step_plot = weight_series.shape[1] // 5
                plt.plot(weight_series[:, ::step_plot], alpha=0.3)
                plt.show()
            coeffs = calculate_coeffs(weight_series, False)
            lr_multipliers[np.abs(coeffs) > tolerance] *= multiplier
            _, x_val, _, y_val = train_test_split(x_train, y_train, test_size=0.1)
            sigma_diag = torch.clamp(sigma_diag, min=1e-10)
            rmse_test, pcip_test, mpiw_test = sample_and_get_metrics(
                x_train, y_train, x_test, y_test, model, theta_swa, sigma_diag, D, K, S, y_scaler
            )
            rmse_val, pcip_val, mpiw_val = sample_and_get_metrics(
                x_train, y_train, x_val, y_val, model, theta_swa, sigma_diag, D, K, S, y_scaler
            )
            round_results.append([rmse_test, pcip_test, mpiw_test, rmse_val, pcip_val, mpiw_val])
        except ValueError:
            pass
    best_ind = np.array(round_results)[:, 3].argmin()
    rmse_test, pcip_test, mpiw_test, rmse_val, pcip_val, mpiw_val = round_results[best_ind]
    print(f"    MCSWAG Test | RMSE: {rmse_test:.3f}, PICP: {pcip_test:.3f}, MPIW:{mpiw_test:.3f} ")
