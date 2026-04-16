from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(real, pred):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    accuracy = 100 - (mae / np.mean(real)) * 100

    return mae, rmse, accuracy