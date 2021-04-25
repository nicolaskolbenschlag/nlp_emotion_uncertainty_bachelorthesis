import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sklearn.isotonic
import scipy.optimize
import config
import typing
import pandas as pd

def uncertainty_measurement_error(real_uncertainty: np.array, predicted_uncertainty: np.ndarray, bins: int = 5) -> float:
    max_uncertainty = predicted_uncertainty.max()

    ume = 0.
    bin_indicies = np.digitize(predicted_uncertainty, np.linspace(predicted_uncertainty.min(), predicted_uncertainty.max(), bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        
        if not mask.sum() > 0:
            continue

        rmse = np.mean(real_uncertainty[mask])
        rmse = np.abs(rmse - 1) / 2

        rmv = np.mean(predicted_uncertainty[mask])
        
        print(f"max_uncertainty: {max_uncertainty}")
        print(f"rmv: {rmv}")
        np.seterr("raise")
        rmv /= max_uncertainty

        ume_ = np.abs(rmv - rmse)
        if rmv != 0.:
            ume_ /= rmv

        ume += ume_

    ume /= bins
    return ume

def stds_coefficient_of_variation(y_pred_var: np.ndarray) -> float:
    mean_of_var = np.mean(y_pred_var)
    cv = np.sqrt(np.power(y_pred_var - mean_of_var, 2).sum() / (len(y_pred_var) - 1)) / mean_of_var
    return cv

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
    
def outputs_mc_dropout(model, test_loader, params, n_ensemble_members = 10):
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
    
    return full_means, full_vars, full_labels, full_subjectivities

def outputs_random(model, test_loader, params, n_ensemble_members = 10):
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
            # NOTE random vector as guessed uncertainty
            vars_ = np.random.uniform(size=means.shape)
            
            full_means.append(means)
            full_vars.append(vars_)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels, full_subjectivities = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels), np.row_stack(full_subjectivities)
    
    return full_means, full_vars, full_labels, full_subjectivities

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

def rolling_correlation_coefficient(y_true: np.array, y_pred: np.array, rolling_window: int = 5) -> np.array:
    error = [
        pd.Series(y_true[i - rolling_window : i]).corr(pd.Series(y_pred[i - rolling_window : i]))
            for i in range(rolling_window, len(y_true) + 1)
        ]
    error = [error[0]] * (rolling_window - 1) + error
    # NOTE [0,0,0].corr([0,0,0]) = nan; therefore interpolate to fill nan
    if np.isnan(error[0]):
        error[0] = 0.
    error = pd.Series(error).interpolate().to_numpy()
    return error

def evaluate_uncertainty_measurement(model, test_loader, val_loader, params, num_bins = 10):
    if params.uncertainty_approach == "monte_carlo_dropout":
        full_means, full_vars, full_labels, full_subjectivities = outputs_mc_dropout(model, test_loader, params)
        method = "MC Dropout"
    
    elif params.uncertainty_approach == "quantile_regression":
        raise NotImplementedError
        full_means, full_vars, full_labels, full_means_val, full_vars_val, full_labels_val = outputs_quantile_regression(model, test_loader, val_loader, params)
        method = "Quantile Regression"
    
    elif params.uncertainty_approach == None:# NOTE random uncertainty generation
        full_means, full_vars, full_labels, full_subjectivities = outputs_random(model, test_loader, params)
        method = "Random"
    
    else:
        raise NotImplementedError

    sbUMEs, pedUMEs, Cvs = [], [], []
    for i in range(full_means.shape[1]):

        # NOTE calculate metrics
        sbUMEs += [uncertainty_measurement_error(full_subjectivities[:,i], full_vars[:,i])]
        pedUMEs += [uncertainty_measurement_error(rolling_correlation_coefficient(full_labels[:,i], full_means[:,i]), full_vars[:,i])]
        Cvs += [stds_coefficient_of_variation(full_vars[:,i])]

        # NOTE plot
        if params.uncertainty_approach != None:
            max_plot = 1000#len(full_labels)
            step_plot = 100
            for j in range(0, max_plot, step_plot):
                plot_confidence(params, full_labels[:,i][j:j+step_plot], full_means[:,i][j:j+step_plot], full_vars[:,i][j:j+step_plot], full_subjectivities[:,i][j:j+step_plot], params.emo_dim_set[i], f"{method} ({j}-{j+step_plot})", test_loader.dataset.partition)
        
    return  sbUMEs, pedUMEs, Cvs