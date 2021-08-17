import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import get_loaders, load_dataset
from definitions import device
from metrics import RMSE
from models import create_model, load_model
from training import get_test_predictions, test_step
from utils import print_losses


class NoisyAdam:
    def __init__(self, model, criterion, N, alpha, beta1, beta2, lambda_, eta, gamma_ex):
        self.criterion = criterion
        self.N = N
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_ = lambda_
        self.eta = eta
        self.gamma_ex = gamma_ex
        self.mu = nn.utils.parameters_to_vector(model.parameters()).detach().cpu().clone()
        self.f = torch.zeros_like(self.mu) + 1e4
        self.m = torch.zeros_like(self.mu)
        self.gamma_in = self.lambda_ / (self.N * self.eta)
        self.gamma = self.gamma_in + self.gamma_ex
        self.k = 0

    def sample_model_weights(self, model):
        #         print(self.mu)
        covarince_m = self.lambda_ / self.N * torch.diag(1 / (self.f + self.gamma_in))
        distribution = torch.distributions.MultivariateNormal(self.mu, covariance_matrix=covarince_m)
        w = distribution.sample()
        return w

    def zero_grad(self, model):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def get_grads(self, model):
        return torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    def get_mu_model(self, model):
        torch.nn.utils.vector_to_parameters(self.mu, model.parameters())
        model.to(device)
        return model

    def sample_posterior(self, x_train, y_train, x_test, y_test, model, S):
        test_predictions_list = []
        for _ in range(S):
            w = self.sample_model_weights(model)
            torch.nn.utils.vector_to_parameters(w, model.parameters())
            model.to(device)
            test_predictions_list.append(get_test_predictions(x_train, y_train, x_test, y_test, model))
        return test_predictions_list

    def step(self, model, batch_x, batch_y):
        self.k += 1

        w = self.sample_model_weights(model)
        torch.nn.utils.vector_to_parameters(w, model.parameters())
        model.to(device)

        self.zero_grad(model)
        pred_y = model(batch_x.to(device))
        loss = self.criterion(pred_y, batch_y)
        loss.backward()
        grads = -self.get_grads(model).detach().cpu().clone()

        v = grads + self.gamma * w

        self.m = self.beta1 * self.m + (1 - self.beta1) * v

        self.f = self.beta2 * self.f + (1 - self.beta2) * grads ** 2

        m_tilde = self.m / (1.0 - self.beta1 ** self.k)

        m_dash = m_tilde / (torch.sqrt(self.f) + self.gamma)

        self.mu = self.mu + self.alpha * m_dash

        return loss.item()


def run_noisy_ADAM(dataset_name, S=500, verbose=False):
    x_train, y_train, x_test, y_test, x_scaler, y_scaler = load_dataset(dataset_name, verbose=False)
    model = create_model(x_train, layer_dims=[50], verbose=False)

    epochs: int = 25
    batch_size: int = 100
    lr: float = 1e-2
    early_stopping_rounds: int = 5
    beta1 = 0.9
    beta2 = 0.9
    lambda_ = 0.01
    eta = 1e6
    gamma_ex = 1e-12

    train_loader, val_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size, val_loader=True)

    criterion = nn.MSELoss()
    best_val_loss = 1e10
    best_epoch = 0

    optimizer = NoisyAdam(model, criterion, x_train.shape[0], lr, beta1, beta2, lambda_, eta, gamma_ex)

    train_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in train_loader]
    val_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in val_loader]
    test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]

    loss_list = [[np.mean(train_losses), np.mean(test_losses), np.mean(val_losses)]]

    if verbose:
        print_losses(0, train_losses, test_losses, val_losses)

    for epoch in range(1, epochs + 1):
        train_losses = [optimizer.step(model, batch_x, batch_y.to(device)) for batch_x, batch_y in train_loader]
        optimizer.get_mu_model(model)
        val_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in val_loader]
        test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
        if verbose:
            print_losses(epoch, train_losses, test_losses, val_losses)
        loss_list.append([np.mean(train_losses), np.mean(test_losses), np.mean(val_losses)])

        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            torch.save(model.state_dict(), f"best_model_weights-{dataset_name}.pth")
            best_val_loss = mean_val_loss
            best_epoch = epoch

        if epoch - best_epoch >= early_stopping_rounds:
            break
    if verbose:
        print(f"Finished Training. Best validation loss: {best_val_loss:.5f} in epoch {best_epoch}")

    model = load_model(model, f"best_model_weights-{dataset_name}.pth", verbose=False)
    y_pred = get_test_predictions(x_train, y_train, x_test, y_test, model)
    if verbose:
        print(f"SGD RMSE: {RMSE(y_pred, y_test, y_scaler):.3f}")
    samples = optimizer.sample_posterior(x_train, y_train, x_test, y_test, model, S)
    samples_array = np.concatenate(samples, axis=1)
    y_pred = samples_array.mean(axis=1, keepdims=True)
    y_l = np.percentile(samples_array, 2.5, axis=1, keepdims=True)
    y_u = np.percentile(samples_array, 97.5, axis=1, keepdims=True)
    pcip = np.mean((y_l < y_test) & (y_test < y_u))
    y_l = y_scaler.inverse_transform(y_l)
    y_u = y_scaler.inverse_transform(y_u)
    print(f"Noisy ADAM Test | RMSE: {RMSE(y_pred, y_test, y_scaler):.3f}, PICP: {pcip:.3f}, MPIW" f""
          f":{np.mean(y_u - y_l):.3f}")
    if verbose:
        plt.plot(np.array(loss_list))
        plt.show()
