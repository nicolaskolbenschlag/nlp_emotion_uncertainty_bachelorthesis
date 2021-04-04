import numpy as np

# NOTE metrics

def expected_normalized_calibration_error(y_real: np.ndarray, y_pred_mean: np.ndarray, y_pred_var: np.ndarray, bins: int = 10) -> float:
    """
    https://arxiv.org/abs/1905.11659
    """
    ence = 0.
    bin_indicies = np.digitize(y_pred_var, np.linspace(y_pred_var.min(), y_pred_var.max(), bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        rmv = np.sqrt(y_pred_var[mask].sum() / mask.sum())
        rmse = np.sqrt(np.power(y_real[mask] - y_pred_mean[mask], 2).sum() / mask.sum())
        ence += np.abs(rmv - rmse) / rmv
    ence /= bins
    return ence

def stds_coefficient_of_variation(y_pred_var: np.ndarray) -> float:
    mean_of_var = np.mean(y_pred_var)
    cv = np.sqrt(np.power(y_pred_var - mean_of_var, 2).sum() / (len(y_pred_var) - 1)) / mean_of_var
    return cv

# NOTE calibration

def calibrate(X_calibrate: np.ndarray, y_calibrate: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    pass


if __name__ == "__main__":
    y_pred = np.array([1,1,1,1,1,1,1,1,1])
    y_var = np.array([1,5,4,2,6,2,9,1,4])
    y_true = np.array([1,1,1,1,1,1,1,1,1])
    e = expected_normalized_calibration_error(y_true, y_pred, y_var, 3)
    print(e)