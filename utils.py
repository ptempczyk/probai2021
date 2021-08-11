import numpy as np


def print_losses(epoch, train_losses, test_losses, val_losses=None):
    if val_losses is None:
        print(f"Epoch: {epoch:3d}, Train loss: {np.mean(train_losses):3.3f}, Test loss: {np.mean(test_losses):3.3f}")
    else:
        print(
            f"Epoch: {epoch:3d}, Train loss: {np.mean(train_losses):3.3f}, Test loss: {np.mean(test_losses):3.3f}, "
            f"Val loss: {np.mean(val_losses):3.3f}"
        )
