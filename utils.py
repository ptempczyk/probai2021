import numpy as np
import pandas as pd
import torch
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


def print_losses(epoch, train_losses, test_losses):
    print(
        f"Epoch: {epoch:3d}, Train loss: {np.mean(train_losses):3.3f}, Test loss: {np.mean(test_losses):3.3f}"
    )


def load_model(model: nn.Module, file_name: str):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model


def get_loaders(
    x_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
    batch_size: int,
):
    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_dataset(dataset_name: str, train_split: float = 0.9, verbose=True):
    df = pd.read_csv(
        DATA_DIR + dataset_name + ".txt", header=None, delim_whitespace=True
    )
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape([-1, 1])
    if verbose:
        print(
            f"dataset: {dataset_name}, rows: {x.shape[0]}, "
            "columns: {x.shape[1]}, range of x: {[x.min(), x.max()]}, "
            "range of y: {[y.min(), y.max()]}"
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
