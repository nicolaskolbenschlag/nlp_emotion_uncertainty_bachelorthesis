import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sklearn.isotonic
import scipy.optimize
import config
import typing

def expected_normalized_calibration_error(y_real: np.ndarray, y_pred_mean: np.ndarray, y_pred_var: np.ndarray, bins: int = 10) -> float:
    """
    https://arxiv.org/abs/1905.11659
    """
    ence = 0.
    bin_indicies = np.digitize(y_pred_var, np.linspace(y_pred_var.min(), y_pred_var.max(), bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        
        if not mask.sum() > 0:
            continue

        rmv = np.sqrt(np.mean(y_pred_var[mask]))
        rmse = np.sqrt(np.mean((y_real[mask] - y_pred_mean[mask]) ** 2))
        
        ence_ = np.abs(rmv - rmse)
        if rmv != 0.:
            ence_ /= rmv

        ence += ence_

    ence /= bins
    return ence

def stds_coefficient_of_variation(y_pred_var: np.ndarray) -> float:
    mean_of_var = np.mean(y_pred_var)
    cv = np.sqrt(np.power(y_pred_var - mean_of_var, 2).sum() / (len(y_pred_var) - 1)) / mean_of_var
    return cv

def calibrate(val_uncalibrated: np.ndarray, val_calibrated: np.ndarray, test_uncalibrated: np.ndarray, method: str = "scalar_fitting") -> np.ndarray:
    if method == "isotonic_regression":
        calibrator = sklearn.isotonic.IsotonicRegression().fit(val_uncalibrated, val_calibrated)
        return calibrator.predict(test_uncalibrated)
    elif method == "scalar_fitting":
        opt = scipy.optimize.minimize_scalar(lambda x: ((val_calibrated - x * val_uncalibrated) ** 2).mean())
        s = opt.x
        return s * test_uncalibrated
    else:
        raise NotImplementedError

def plot_confidence(params, labels: np.ndarray, pred_mean: np.ndarray, pred_confidence: np.ndarray, emo_dim: str, title: str, partition: str, timesteps: typing.Tuple[int, int] = (0, 100)) -> None:
    labels = labels[timesteps[0] : timesteps[1]]
    pred_mean = pred_mean[timesteps[0] : timesteps[1]]
    pred_confidence = pred_confidence[timesteps[0] : timesteps[1]]

    time = range(len(labels))
    plt.figure(figsize=(20, 10))
    plt.plot(time, labels, "red", label="target")
    plt.plot(time, pred_mean, "blue", label="prediction")
    plt.fill_between(time, pred_mean - pred_confidence, pred_mean + pred_confidence, color="lightblue", alpha=.5)

    plt.title(f"{title} [{partition}]", fontsize=24)
    plt.legend(prop={"size": 24})
    plt.xlabel("time", fontsize=24)
    plt.ylabel(emo_dim, fontsize=24)
    # plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    dir = config.PREDICTION_FOLDER
    if params.save_dir is not None: dir = os.path.join(dir, params.save_dir)
    dir = os.path.join(dir, "img_calibration")
    if not os.path.exists(dir): os.mkdir(dir)
    dir = os.path.join(dir, emo_dim)
    if not os.path.exists(dir): os.mkdir(dir)
    dir = os.path.join(dir, params.uncertainty_approach)
    if not os.path.exists(dir): os.mkdir(dir)
    dir = os.path.join(dir, "seed" + str(params.current_seed))
    if not os.path.exists(dir): os.mkdir(dir)
    
    title = title.replace(" ", "_")
    plt.savefig(os.path.join(dir, f"{title}.jpg"))
    plt.close()

def outputs_mc_dropout(model, test_loader, val_loader, params, n_ensemble_members = 10):
    # NOTE make predictions for uncalibrated scores and as features to become calibrated later
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
            
    # NOTE prepare recalibration (by providing validation data as training data for calibration-model)
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
    
    return full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val

def outputs_quantile_regression(model, test_loader, val_loader, params):
    full_means, full_vars, full_labels = [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens).cpu().detach().squeeze(0).numpy()
            means = preds[:, 1:2]
            
            # NOTE use difference between upper and lower quantile as measurement for uncalibrated confidence
            vars = preds[:, 2:3] - preds[:, 0:1]
            vars = np.abs(vars)
            
            full_means.append(means)
            full_vars.append(vars)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels)
        
    full_means_val, full_vars_val, full_labels_val = [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(val_loader, 1):
            features, feature_lens, labels, meta = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = model(features, feature_lens).cpu().detach().squeeze(0).numpy()
            means = preds[:, 1:2]
            
            vars = preds[:, 2:3] - preds[:, 0:1]
            vars = np.abs(vars)

            full_means_val.append(means)
            full_vars_val.append(vars)
            full_labels_val.append(labels.cpu().detach().squeeze(0).numpy())

        full_means_val, full_vars_val, full_labels_val = np.row_stack(full_means_val), np.row_stack(full_vars_val), np.row_stack(full_labels_val)
    
    return full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val

def evaluate_calibration(model, test_loader, val_loader, params, num_bins = 10):
    if params.uncertainty_approach == "monte_carlo_dropout":
        full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val = outputs_mc_dropout(model, test_loader, val_loader, params)
        title = "MC Dropout"
    elif params.uncertainty_approach == "quantile_regression":
        full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val = outputs_quantile_regression(model, test_loader, val_loader, params)
        title = "Quantile Regression"
    else:
        raise NotImplementedError

    # NOTE calculate metrics
    ENCEs_uncal, ENCEs_cal, Cvs_uncal, Cvs_cal = [], [], [], []
    for i in range(full_means.shape[1]):

        # NOTE measurement of metrics uncalibrated
        ence_uncalibrated = expected_normalized_calibration_error(full_labels[:, i], full_means[:, i], full_vars[:, i], num_bins)
        cv_uncalibrated = stds_coefficient_of_variation(full_vars[:, i])
        ENCEs_uncal.append(ence_uncalibrated)
        Cvs_uncal.append(cv_uncalibrated)

        plot_confidence(params, full_labels[:, i], full_means[:, i], full_vars[:, i], params.emo_dim_set[i], title + " UNCALIBRATED", test_loader.dataset.partition)

        # NOTE rmse of validaiton set as calibration target
        rmse_val = np.sqrt(np.mean((full_labels_val[:, i] - full_means_val[:, i]) ** 2))
        # NOTE recalibration
        full_vars_calibrated = calibrate(full_vars_val[:, i], rmse_val, full_vars[:, i])

        # NOTE measurement of metrics calibrated
        ence_calibrated = expected_normalized_calibration_error(full_labels[:, i], full_means[:, i], full_vars_calibrated, num_bins)
        cv_calibrated = stds_coefficient_of_variation(full_vars_calibrated)
        ENCEs_cal.append(ence_calibrated)
        Cvs_cal.append(cv_calibrated)
        
        plot_confidence(params, full_labels[:, i], full_means[:, i], full_vars_calibrated, params.emo_dim_set[i], title + " CALIBRATED", test_loader.dataset.partition)
        
    return ENCEs_uncal, ENCEs_cal, Cvs_uncal, Cvs_cal