import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal

from data import get_loaders, load_dataset
from definitions import LR_SWAG
from metrics import RMSE
from models import create_model, load_model
from training import test_step, train_step, get_test_predictions, train_model
from utils import print_losses


def train_SWAG(
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


def calculate_sigma(sigma_diag: torch.Tensor, D: torch.Tensor, K: int):
    assert K >= 2
    return 0.5 * torch.diag(sigma_diag) + 0.5 * torch.mm(D, D.T) / (K + 1)


def sample_posterior(theta_swa: torch.Tensor, sigma_diag: torch.Tensor, D: torch.Tensor, K: int):
    assert K >= 2
    mu = theta_swa
    sigma = calculate_sigma(sigma_diag, D, K)
    distribution = MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return distribution.sample()


def sample_from_SWAG(x_train, y_train, x_test, y_test, model, theta_swa, sigma_diag, D, K, S):
    test_predictions_list = []
    for _ in range(S):
        sampled_weights = sample_posterior(theta_swa, sigma_diag, D, K)
        torch.nn.utils.vector_to_parameters(sampled_weights, model.parameters())
        test_predictions_list.append(get_test_predictions(x_train, y_train, x_test, y_test, model))
    return test_predictions_list


def run_SWAG(dataset_name, K=10, S=500, weight_decay=1e-6):
    print("=" * 88)
    x_train, y_train, x_test, y_test, _, _ = load_dataset(dataset_name)
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
    y_pred = get_test_predictions(x_train, y_train, x_test, y_test, model)
    print(f"SGD RMSE: {RMSE(y_pred, y_test):.3f}")
    SWAG_lr = LR_SWAG[dataset_name]
    print(f"SWAG_lr: {SWAG_lr: 0.5f}")
    model = load_model(model, f"best_model_weights-{dataset_name}.pth", verbose=False)
    theta_swa, sigma_diag, D, thetas = train_SWAG(
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        K,
        verbose=False,
        lr=SWAG_lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=50,
    )
    sigma_diag = torch.clamp(sigma_diag, min=1e-10)
    samples = sample_from_SWAG(x_train, y_train, x_test, y_test, model, theta_swa, sigma_diag, D, K, S)
    samples_array = np.concatenate(samples, axis=1)
    y_pred = samples_array.mean(axis=1, keepdims=True)
    y_l = np.percentile(samples_array, 2.5, axis=1, keepdims=True)
    y_u = np.percentile(samples_array, 97.5, axis=1, keepdims=True)
    print(
        f"RMSE: {RMSE(y_pred, y_test):.3f}, PICP: {np.mean((y_l < y_test) & (y_test < y_u)):.3f}, MPIW:{np.mean(y_u - y_l):.3f}"
    )
