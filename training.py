import numpy as np
import torch
from torch import nn

from data import get_loaders
from utils import print_losses


def train_step(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
):
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test_step(batch_x: torch.Tensor, batch_y: torch.Tensor, model: nn.Module, criterion):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    return loss.item()


def train_model(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    model: nn.Module,
    dataset_name,
    epochs: int = 2000,
    batch_size: int = 100,
    lr: float = 1e-2,
    early_stopping_rounds: int = 5,
    verbose: bool = True,
    momentum=0,
    weight_decay=0,
):
    train_loader, val_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size, val_loader=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = 1e10
    best_epoch = 0

    train_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in train_loader]
    val_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in val_loader]
    test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
    if verbose:
        print_losses(0, train_losses, test_losses, val_losses)

    for epoch in range(1, epochs + 1):
        train_losses = [train_step(batch_x, batch_y, model, optimizer, criterion) for batch_x, batch_y in train_loader]
        val_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in val_loader]
        test_losses = [test_step(batch_x, batch_y, model, criterion) for batch_x, batch_y in test_loader]
        if verbose:
            print_losses(0, train_losses, test_losses, val_losses)
        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            torch.save(model.state_dict(), f"best_model_weights-{dataset_name}.pth")
            best_val_loss = mean_val_loss
            best_epoch = epoch

        if epoch - best_epoch >= early_stopping_rounds:
            break
    print(f"Finished Training. Best validation loss: {best_val_loss:.5f} in epoch {best_epoch}")


def get_test_predictions(
    x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, model: nn.Module, batch_size: int = 100
):
    train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test, batch_size)
    test_predictions = [model(batch_x).cpu().detach().numpy() for batch_x, _ in test_loader]
    return np.concatenate(test_predictions)
