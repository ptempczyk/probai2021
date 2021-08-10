import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "../dt8122-2021/datasets/"
DATASETS = """boston_housing
concrete
energy_heating_load
kin8nm
naval_compressor_decay
power
protein
wine
yacht
year_prediction_msd""".split()

LR_SWAG = {
    "boston_housing": 0.01,
    "concrete": 0.2,  # maybe try more
    "energy_heating_load": 0.2,
    "kin8nm": 0.1,
    "naval_compressor_decay": 0.1,  # maybe try more
    "power": 0.2,  # maybe try more
    "protein": 1,
    "wine": 1,
    "yacht": 1,
    "year_prediction_msd": 1,
}


def print_losses(epoch, train_losses, test_losses, val_losses=None):
    if val_losses is None:
        print(f"Epoch: {epoch:3d}, Train loss: {np.mean(train_losses):3.3f}, Test loss: {np.mean(test_losses):3.3f}")
    else:
        print(f"Epoch: {epoch:3d}, Train loss: {np.mean(train_losses):3.3f}, Test loss: {np.mean(test_losses):3.3f}, "
              f"Val loss: {np.mean(val_losses):3.3f}")


def load_model(model: nn.Module, file_name: str, verbose: bool = True):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    if verbose:
        print(f"Model {file_name} loaded.")
    return model


def get_loaders(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    batch_size: int,
    val_loader: bool = False
):
    if not val_loader:
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
        test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


def load_dataset(dataset_name: str, train_split: float = 0.9, verbose=True):
    df = pd.read_csv(DATA_DIR + dataset_name + ".txt", header=None, delim_whitespace=True)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape([-1, 1])
    if verbose:
        print(
            f"dataset: {dataset_name}, rows: {x.shape[0]}, "
            f"columns: {x.shape[1]}, range of x: {[x.min(), x.max()]}, "
            f"range of y: {[y.min(), y.max()]}"
        )
    x_scaler = StandardScaler()
    x_scaler.fit(x)
    x = x_scaler.transform(x)
    y_scaler = StandardScaler()
    y_scaler.fit(y)
    y = y_scaler.transform(y)
    length = len(y)
    split = int(length * train_split)
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]
    return x_train, y_train, x_test, y_test, x_scaler, y_scaler


def create_model(x_train: np.array, layer_dims: list = None, verbose: bool = True):
    if layer_dims is None:
        layer_dims = [100, 50]
    n_features = x_train.shape[1]
    layer_dims = [n_features] + layer_dims
    layers = []
    for i in range(1, len(layer_dims)):
        layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
        layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(layer_dims[-1], 1))
    model = nn.Sequential(*layers)
    if verbose:
        print(model)
    return model


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
    best_val_loss = 1000000.0
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
    return theta, sigma_diag, D
