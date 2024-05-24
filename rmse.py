import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    mse = np.mean((pred - tar) ** 2)
    rmse = np.sqrt(mse)
    return rmse
