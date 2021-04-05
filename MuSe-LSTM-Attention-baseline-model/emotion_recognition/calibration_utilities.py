import numpy as np
import torch
import sklearn.isotonic

def expected_normalized_calibration_error(y_real: np.ndarray, y_pred_mean: np.ndarray, y_pred_var: np.ndarray, bins: int = 10) -> float:
    """
    https://arxiv.org/abs/1905.11659
    """
    ence = 0.
    bin_indicies = np.digitize(y_pred_var, np.linspace(y_pred_var.min(), y_pred_var.max(), bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        # rmv = np.sqrt(y_pred_var[mask].sum() / mask.sum())
        rmv = np.sqrt(np.mean(y_pred_var[mask]))
        # rmse = np.sqrt(np.power(y_real[mask] - y_pred_mean[mask], 2).sum() / mask.sum())
        rmse = np.sqrt(np.mean((y_real[mask] - y_pred_mean[mask]) ** 2))
        ence += np.abs(rmv - rmse) / rmv
    ence /= bins
    return ence

def stds_coefficient_of_variation(y_pred_var: np.ndarray) -> float:
    mean_of_var = np.mean(y_pred_var)
    cv = np.sqrt(np.power(y_pred_var - mean_of_var, 2).sum() / (len(y_pred_var) - 1)) / mean_of_var
    # cv = np.sqrt(np.mean((y_pred_var - mean_of_var) ** 2)) / np.mean(y_pred_var)
    return cv

def calibrate(val_uncalibrated: np.ndarray, val_calibrated: np.ndarray, test_uncalibrated: np.ndarray) -> np.ndarray:
    calibrator = sklearn.isotonic.IsotonicRegression().fit(val_uncalibrated, val_calibrated)
    test_pred = calibrator.predict(test_uncalibrated)
    return test_pred

def evaluate_mc_dropout_calibration(model, test_loader, val_loader, params, n_ensemble_members = 10):
    num_bins = 10

    model.train()
    full_means, full_vars, full_labels = [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            vars = np.var(preds, axis=0)
            
            full_means.append(means)
            full_vars.append(vars)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

    full_means, full_vars, full_labels = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels)
        
    # NOTE measurement of metrics uncalibrated
    ence_uncalibrated = expected_normalized_calibration_error(full_labels, full_means, full_vars, num_bins)
    cv_uncalibrated = stds_coefficient_of_variation(full_vars)
        
    # NOTE prepare recalibration
    full_means_val, full_vars_val, full_labels_val = [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(val_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            vars = np.var(preds, axis=0)

            full_means_val.append(means)
            full_vars_val.append(vars)
            full_labels_val.append(labels.cpu().detach().squeeze(0).numpy())

    full_means_val, full_vars_val, full_labels_val = np.row_stack(full_means_val), np.row_stack(full_vars_val), np.row_stack(full_labels_val)

    # NOTE rmse of validaiton set as calibration target
    rmse_val = np.sqrt(np.mean((full_labels_val - full_means_val) ** 2))
    # NOTE recalibration
    full_vars_calibrated = calibrate(full_vars_val, rmse_val, full_vars)

    # NOTE measurement of metrics calibrated
    ence_calibrated = expected_normalized_calibration_error(full_labels, full_means, full_vars_calibrated, num_bins)
    cv_calibrated = stds_coefficient_of_variation(full_vars_calibrated)

    return ence_uncalibrated, cv_uncalibrated, ence_calibrated, cv_calibrated

if __name__ == "__main__":
    y_pred = np.array([1,1,1,1,1,1,1,1,1])
    y_var = np.array([1,5,4,2,6,2,9,1,4])
    y_true = np.array([1,1,1,1,1,1,1,1,1])
    e = expected_normalized_calibration_error(y_true, y_pred, y_var, 3)
    print(e)