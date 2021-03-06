import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sklearn.isotonic
import scipy.optimize
import config
import typing
import pandas as pd

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

def subjectivity_based_ence(y_real_subjecivities: np.ndarray, y_pred_var: np.ndarray, bins: int = 5) -> float:
    max_uncertainty = y_pred_var.max()

    ence = 0.
    bin_indicies = np.digitize(y_pred_var, np.linspace(y_pred_var.min(), y_pred_var.max(), bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        
        if not mask.sum() > 0:
            continue

        # NOTE use mae between predicted quantification of uncertainty (=y_pred_var, so rmv) and real uncertainty quantified by subjectivity among annotations over time (=y_real_subjecivities), to evaluate quality of uncertainty quantification, instead of rmse between prediction and label on a single timestep
        # rmse = np.sqrt(np.mean((y_real[mask] - y_pred_mean[mask]) ** 2))# NOTE this line is usually in ENCE
        rmse = np.mean(y_real_subjecivities[mask])
        # NOTE both pearson correlation and CCC range from -1 to +1; we want large uncertainty quantification from the model if true subjectivity goes towards -1 and lower if toward +1; therefore...
        # NOTE now 0 is perfectly corrlelated (so zero subjectivity), so we expect small rmv, and +1 (because of abs) is huge true subjectivity and we want rmv to be large
        rmse = np.abs(rmse - 1) / 2

        # rmv = np.sqrt(np.mean(y_pred_var[mask]))
        rmv = np.mean(y_pred_var[mask])
        
        # NOTE scale rmv (=measurement of predicted uncertainty) to [0,1], because our true subjectivity (here: rmse) is also limited to [0,1]
        print(f"max_uncertainty: {max_uncertainty}")
        print(f"rmv: {rmv}")
        np.seterr("raise")
        rmv /= max_uncertainty

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

def calibrate(val_uncalibrated: np.array, val_calibrated: np.array, test_uncalibrated: np.array, method: str = "scalar_fitting") -> np.array:
    if method == "isotonic_regression":
        calibrator = sklearn.isotonic.IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(val_uncalibrated, val_calibrated)# NOTE increasing="auto"?
        return calibrator.predict(test_uncalibrated)
    elif method == "scalar_fitting":
        opt = scipy.optimize.minimize_scalar(lambda x: ((val_calibrated - x * val_uncalibrated) ** 2).mean())
        s = opt.x
        return s * test_uncalibrated
    else:
        raise NotImplementedError

def plot_confidence(params, labels: np.ndarray, pred_mean: np.ndarray, pred_confidence: np.ndarray, subjectivety: np.ndarray, emo_dim: str, title: str, partition: str) -> None:
    time = range(len(labels))
    fig, axs = plt.subplots(2, 1, figsize=(20,10))

    fig.suptitle(f"{title} [{partition}]", fontsize=24)

    axs[0].plot(time, labels, "red", label="target")
    axs[0].plot(time, pred_mean, "blue", label="prediction")
    axs[0].fill_between(time, pred_mean - pred_confidence, pred_mean + pred_confidence, color="lightblue", alpha=.5)
    
    axs[0].legend(prop={"size": 24})

    axs[0].set_xlabel("time", fontsize=24)
    axs[0].set_ylabel(emo_dim, fontsize=24)

    axs[1].plot(time, subjectivety, "orange")

    axs[1].set_xlabel("time", fontsize=24)
    axs[1].set_ylabel("true confidence", fontsize=24)
    axs[1].set_ylim(-1., 1.)

    for tick in axs[0].xaxis.get_major_ticks() + axs[0].yaxis.get_major_ticks() + axs[1].xaxis.get_major_ticks() + axs[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

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
    fig.savefig(os.path.join(dir, f"{title}.jpg"))
    plt.close()
    
def outputs_mc_dropout(model, test_loader, val_loader, params, n_ensemble_members = 10):
    # NOTE make predictions for uncalibrated scores and as features to become calibrated later
    model.train()
    full_means, full_vars, full_labels, full_subjectivities = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
                subjectivities = subjectivities.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            vars_ = np.var(preds, axis=0)
            
            full_means.append(means)
            full_vars.append(vars_)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels, full_subjectivities = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels), np.row_stack(full_subjectivities)
            
    # NOTE prepare recalibration (by providing validation data as training data for calibration-model)
    full_means_val, full_vars_val, full_labels_val, full_subjectivities_val = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(val_loader, 1):
            features, feature_lens, labels, meta, subjectivities = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
                subjectivities = subjectivities.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            vars_ = np.var(preds, axis=0)

            full_means_val.append(means)
            full_vars_val.append(vars_)
            full_labels_val.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities_val.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means_val, full_vars_val, full_labels_val, full_subjectivities_val = np.row_stack(full_means_val), np.row_stack(full_vars_val), np.row_stack(full_labels_val), np.row_stack(full_subjectivities_val)
    
    return full_means, full_vars, full_labels, full_subjectivities, full_means_val, full_vars_val, full_labels_val, full_subjectivities_val

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
        full_means, full_vars, full_labels, full_subjectivities, full_means_val, full_vars_val, full_labels_val, full_subjectivities_val = outputs_mc_dropout(model, test_loader, val_loader, params)
        method = "MC Dropout"
    elif params.uncertainty_approach == "quantile_regression":
        full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val = outputs_quantile_regression(model, test_loader, val_loader, params)
        method = "Quantile Regression"
    else:
        raise NotImplementedError

    # NOTE calculate metrics
    ENCEs_uncal, ENCEs_cal, Cvs_uncal, Cvs_cal = [], [], [], []
    for i in range(full_means.shape[1]):

        # NOTE measurement of metrics uncalibrated
        # ence_uncalibrated = expected_normalized_calibration_error(full_labels[:,i], full_means[:,i], full_vars[:,i], num_bins)
        ence_uncalibrated = subjectivity_based_ence(full_subjectivities[:,i], full_vars[:,i])
        print("ence_uncalibrated:", ence_uncalibrated)
        cv_uncalibrated = stds_coefficient_of_variation(full_vars[:,i])
        ENCEs_uncal.append(ence_uncalibrated)
        Cvs_uncal.append(cv_uncalibrated)

        max_plot = 1000#len(full_labels)
        step_plot = 100
        for j in range(0, max_plot, step_plot):
            plot_confidence(params, full_labels[:,i][j:j+step_plot], full_means[:,i][j:j+step_plot], full_vars[:,i][j:j+step_plot], full_subjectivities[:,i][j:j+step_plot], params.emo_dim_set[i], f"{method} UNCALIBRATED ({j}-{j+step_plot})", test_loader.dataset.partition)

        ##########################
        # NOTE rmse of validaiton set as calibration target
        # rmse_val = np.sqrt(np.mean((full_labels_val[:,i] - full_means_val[:,i]) ** 2))        
        # NOTE recalibration
        # full_vars_calibrated = calibrate(full_vars_val[:,i], rmse_val, full_vars[:,i])
        ##########################
        # NOTE adjust calibration mechanism: rmse doesn't mak sense anymore as calibration target, because true uncertainty is measured as subjectivity now        
        opt = scipy.optimize.minimize_scalar(lambda x: subjectivity_based_ence(full_subjectivities_val[:,i], full_vars_val[:,i] * x))
        print(f"Calibration scalar: {opt.x}")
        full_vars_calibrated = full_vars[:,i] * opt.x
        ##########################

        # NOTE measurement of metrics calibrated
        # ence_calibrated = expected_normalized_calibration_error(full_labels[:,i], full_means[:,i], full_vars_calibrated, num_bins)
        ence_calibrated = subjectivity_based_ence(full_subjectivities[:,i], full_vars_calibrated)
        cv_calibrated = stds_coefficient_of_variation(full_vars_calibrated)
        ENCEs_cal.append(ence_calibrated)
        Cvs_cal.append(cv_calibrated)
        
        for j in range(0, max_plot, step_plot):
            plot_confidence(params, full_labels[:,i][j:j+step_plot], full_means[:,i][j:j+step_plot], full_vars_calibrated[j:j+step_plot], full_subjectivities[:,i][j:j+step_plot], params.emo_dim_set[i], f"{method} CALIBRATED ({j}-{j+step_plot})", test_loader.dataset.partition)
        
    return ENCEs_uncal, ENCEs_cal, Cvs_uncal, Cvs_cal