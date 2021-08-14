import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from definitions import DATA_DIR


def get_loaders(
    x_train_orig: np.array, y_train_orig: np.array, x_test: np.array, y_test: np.array, batch_size: int, val_loader:
        bool = False
):
    if not val_loader:
        train_ds = TensorDataset(torch.from_numpy(x_train_orig).float(), torch.from_numpy(y_train_orig).float())
        test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train_orig, y_train_orig, test_size=0.1)
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


def train_test_split(x, y, test_size=0.1):
    length = x.shape[0]
    train_split = 1 - test_size
    split = int(length * train_split)
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]
    return x_train, x_test, y_train, y_test
