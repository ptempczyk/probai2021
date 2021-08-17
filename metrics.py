import numpy as np
from sklearn.preprocessing import StandardScaler


def RMSE(y_1: np.array, y_2: np.array, y_scaler: StandardScaler):
    y_1_resc = y_scaler.inverse_transform(y_1)
    y_2_resc = y_scaler.inverse_transform(y_2)
    return np.sqrt(np.mean((y_1_resc - y_2_resc) ** 2))
