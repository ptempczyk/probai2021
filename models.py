import numpy as np
import torch
from torch import nn


def load_model(model: nn.Module, file_name: str, verbose: bool = True):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    if verbose:
        print(f"Model {file_name} loaded.")
    return model


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
