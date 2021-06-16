import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import config
import typing
import pandas as pd

import calibration_utilities_deprecated
import uncertainty_utilities_global

def uncertainty_measurement_error_divide_by_predicted_uncertainty(real_uncertainty: np.array, predicted_uncertainty: np.ndarray, bins: int = 5) -> float:
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

        np.seterr("raise")# NOTE sometimes in next line numpy error: 'invalid value encountered'
        rmv /= max_uncertainty

        ume_ = np.abs(rmv - rmse)
        if rmv != 0.:
            ume_ /= rmv

        ume += ume_

    ume /= bins
    return ume

def uncertainty_measurement_error(real_uncertainty: np.array, predicted_uncertainty: np.ndarray, bins: int = 20) -> float:
    predicted_uncertainty /= predicted_uncertainty.max()

    ume = 0.
    bin_indicies = np.digitize(real_uncertainty, np.linspace(-1., 1., bins))
    for j in range(1, bins + 1):
        mask = bin_indicies == j
        if not mask.sum() > 0:
            continue

        rmse = np.mean(real_uncertainty[mask])
        rmse = np.abs(rmse - 1) / 2

        rmv = np.mean(predicted_uncertainty[mask])
        
        ume_ = np.abs(rmv - rmse) / rmse
        ume += ume_

    ume /= bins
    return ume

def UME_abs_experimental(real_uncertainty: np.array, predicted_uncertainty, ume_rolling_scaling_window: int) -> float:
    real_uncertainty = np.abs(real_uncertainty - 1) / 2

    def rolling_scaling(array: np.array, window: int) -> np.array:
        out = np.empty_like(array)
        for i in range(0, len(array), window):
            tmp = array[i : min(i + window, len(array))]
            tmp -= tmp.min()
            if tmp.max() != 0: tmp = tmp / tmp.max()
            out[i : min(i + window, len(array))] = tmp
        return out

    if ume_rolling_scaling_window is None:
        real_uncertainty -= real_uncertainty.min()
        real_uncertainty = real_uncertainty / real_uncertainty.max()
        predicted_uncertainty -= predicted_uncertainty.min()
        predicted_uncertainty = predicted_uncertainty / predicted_uncertainty.max()

    else:
        real_uncertainty = rolling_scaling(real_uncertainty, ume_rolling_scaling_window)
        predicted_uncertainty = rolling_scaling(predicted_uncertainty, ume_rolling_scaling_window)

    return np.mean(np.abs(predicted_uncertainty - real_uncertainty))

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
    
def outputs_mc_dropout(model, test_loader, params, n_ensemble_members = 5):
    model.train()
    full_means, full_vars, full_labels, full_subjectivities = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities, _ = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
                subjectivities = subjectivities.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            
            # vars_ = np.var(preds, axis=0)
            rolling_window = 3
            vars_ = []
            for dim in range(means.shape[1]):
                vars_dim = []
                for k, p1 in enumerate(preds):
                    for p2 in preds[k+1:]:
                        corr = [
                            pd.Series(p1[:,dim][i - rolling_window : i]).corr(pd.Series(p2[:,dim][i - rolling_window : i]))
                            for i in range(rolling_window, len(p1) + 1)
                            ]
                        corr = [corr[0]] * (rolling_window - 1) + corr
                        if np.isnan(corr[0]): corr[0] = 0.
                        corr = pd.Series(corr).interpolate()
                        vars_dim += [corr]
                vars_dim = np.stack(vars_dim).mean(axis=0)
                vars_ += [vars_dim]
            vars_ = np.column_stack(vars_)
            vars_ = np.abs(vars_ - 1) / 2
            assert vars_.shape == means.shape
            
            full_means.append(means)
            full_vars.append(vars_)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels, full_subjectivities = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels), np.row_stack(full_subjectivities)
    
    return full_means, full_vars, full_labels, full_subjectivities

def outputs_random(model, test_loader, params):
    model.eval()
    full_means, full_vars, full_labels, full_subjectivities = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities, _ = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
                subjectivities = subjectivities.cuda()
            preds = model(features, feature_lens).cpu().detach().squeeze(0).numpy()
            means = preds
            # NOTE random vector as predicted/guessed uncertainty
            vars_ = np.random.uniform(size=means.shape)
            
            full_means.append(means)
            full_vars.append(vars_)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels, full_subjectivities = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels), np.row_stack(full_subjectivities)
    
    return full_means, full_vars, full_labels, full_subjectivities

def outputs_quantile_regression(model, test_loader, params):
    full_means, full_vars, full_labels, full_subjectivities = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities, _ = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
                subjectivities = subjectivities.cuda()
            preds = model(features, feature_lens).cpu().detach().squeeze(0).numpy()
            
            assert preds.shape[1] == 3, "currently only one emo. dim. with 3 quantiles supported"
            means = preds[:,1:2]
            
            rolling_window = 3
            vars_ = [
                pd.Series(preds[:,0][i - rolling_window : i]).corr(pd.Series(preds[:,2][i - rolling_window : i]))
                for i in range(rolling_window, len(preds) + 1)
            ]
            vars_ = [vars_[0]] * (rolling_window - 1) + vars_
            if np.isnan(vars_[0]): vars_[0] = 0.
            vars_ = pd.Series(vars_).interpolate()
            vars_ = np.array(vars_)[:,np.newaxis]
            vars_ = np.abs(vars_ - 1) / 2
            assert vars_.shape == means.shape
            
            full_means.append(means)
            full_vars.append(vars_)
            full_labels.append(labels.cpu().detach().squeeze(0).numpy())
            full_subjectivities.append(subjectivities.cpu().detach().squeeze(0).numpy())

        full_means, full_vars, full_labels, full_subjectivities = np.row_stack(full_means), np.row_stack(full_vars), np.row_stack(full_labels), np.row_stack(full_subjectivities)
    
    return full_means, full_vars, full_labels, full_subjectivities

def rolling_correlation_coefficient(y_true: np.array, y_pred: np.array, rolling_window: int) -> np.array:
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

def ccc_score(x: np.array, y: np.array) -> float:
    x_mean, y_mean = np.mean(x), np.mean(y)
    cov_mat = np.cov(x, y)
    covariance = cov_mat[0,1]
    x_var, y_var = cov_mat[0,0], cov_mat[1,1]
    ccc = 2. * covariance / (x_var + y_var + (x_mean - y_mean) ** 2)
    return ccc

def subjectivity_vs_rolling_correlation_error(subjectivities: np.ndarray, labels: np.ndarray, means: np.ndarray):
    
    SvCs = []
    for i in range(subjectivities.shape[1]):
        tmp = {}
        for window in [3,5,7,10]:
            correlation_error = rolling_correlation_coefficient(labels[:,i], means[:,i], window)
            tmp[window] = ccc_score(subjectivities[:,i], correlation_error)
        SvCs += [tmp]
    return SvCs

def calculate_uncertainty_metrics(params, labels: np.ndarray, means: np.ndarray, vars_: np.ndarray, subjectivities: np.ndarray, method: str, partition: str, plot: bool = True, benchmark: bool = False):
    sbUMEs, pebUMEs, Cvs = [], [], []
    for i in range(means.shape[1]):        

        tmp_0_sbUME = {}
        tmp_0_pebUME = {}

        # NOTE window that determines how UME is calculated
        for scaling_window in [None, 10, 200, 500]:

            if not benchmark:
                tmp_0_sbUME[scaling_window] = UME_abs_experimental(subjectivities[:,i], vars_[:,i], scaling_window)
            else:
                tmp_0_sbUME[scaling_window] = UME_abs_experimental(subjectivities[:,i], np.random.normal(subjectivities[:,i].mean(), subjectivities[:,i].std(), subjectivities[:,i].shape), scaling_window)

            tmp_1_pebUME = {}

            # NOTE window that defines rolling correlation error
            for window in [3,5,7,10]:
                err = rolling_correlation_coefficient(labels[:,i], means[:,i], window)
                if not benchmark:
                    pebUME = UME_abs_experimental(err, vars_[:,i], scaling_window)
                else:
                    pebUME = UME_abs_experimental(err, np.random.normal(err.mean(), err.std(), err.shape), scaling_window)
                tmp_1_pebUME[window] = pebUME
            
            tmp_0_pebUME[scaling_window] = tmp_1_pebUME
        
        sbUMEs += [tmp_0_sbUME]
        pebUMEs += [tmp_0_pebUME]
        
        if not benchmark:
            Cvs += [{
                "predicted uncertainty": stds_coefficient_of_variation(vars_[:,i]),
                "true subjectivity": stds_coefficient_of_variation(subjectivities[:,i]),
                "true rolling error 3": stds_coefficient_of_variation(rolling_correlation_coefficient(labels[:,i], means[:,i], 3)),
            }]

        if plot:
            assert not benchmark
            max_plot = 1000#len(labels)
            step_plot = 100
            for j in range(0, max_plot, step_plot):
                plot_confidence(
                    params,
                    labels[:,i][j:j+step_plot],
                    means[:,i][j:j+step_plot],
                    vars_[:,i][j:j+step_plot],
                    subjectivities[:,i][j:j+step_plot],
                    params.emo_dim_set[i],
                    f"{method} ({j}-{j+step_plot})",
                    partition)
    
    return sbUMEs, pebUMEs, Cvs

def evaluate_uncertainty_measurement(model, test_loader, params, val_loader):
    print("-" * 20 + "TEST" + "-" * 20)
    evaluate_uncertainty_measurement_help(model, test_loader, params, val_loader)
    
    print("-" * 20 + "DEVEL" + "-" * 20)
    evaluate_uncertainty_measurement_help(model, val_loader, params, val_loader)

def evaluate_uncertainty_measurement_help(model, test_loader, params, val_loader):
    if params.uncertainty_approach == "monte_carlo_dropout":
        prediction_fn = outputs_mc_dropout
        method = "MC Dropout"
    
    elif params.uncertainty_approach == "quantile_regression":
        prediction_fn = outputs_quantile_regression
        method = "Tilt by Correlation"
    
    # NOTE random uncertainty generation
    elif params.uncertainty_approach == None:
        prediction_fn = outputs_random
        method = "Random"
        raise NotImplementedError
    
    else:
        raise NotImplementedError
    
    data_to_store = {}

    full_means, full_vars, full_labels, full_subjectivities = prediction_fn(model, test_loader, params)
    data_to_store["subjectivities"] = full_subjectivities
    data_to_store["subjectivities_pred"] = full_vars
    
    sbUMEs, pebUMEs, Cvs = calculate_uncertainty_metrics(params, full_labels, full_means, full_vars, full_subjectivities, method + "(uncal.)", test_loader.dataset.partition, params.uncertainty_approach != None)
    print(f"UNCALIBRATED\nsbUMEs: {sbUMEs}\npebUMES{pebUMEs}\nCvs: {Cvs}")

    # NOTE compare subjectivity and rolling correlation error
    SvCs = subjectivity_vs_rolling_correlation_error(full_subjectivities, full_labels, full_means)
    print(f"Subjectivity vs. roll.-corr.-coef.: {SvCs}")

    full_means_val, full_vars_val, full_labels_val, full_subjectivities_val = prediction_fn(model, val_loader, params)
    data_to_store["subjectivities_val"] = full_subjectivities_val
    data_to_store["subjectivities_pred_val"] = full_vars_val
    
    for calibration_target in ["subjectivity", "rolling_error_3", "rolling_error_5"]:
    
        full_vars_calibrated = np.empty_like(full_vars)
        for i in range(full_means.shape[1]):
            calibration_features = full_vars_val[:,i]
            
            if calibration_target == "subjectivity":
                true_uncertainty = full_subjectivities_val[:,i]
            elif calibration_target == "rolling_error_3":
                true_uncertainty = rolling_correlation_coefficient(full_labels_val[:,i], full_means_val[:,i], 3)
            elif calibration_target == "rolling_error_5":
                true_uncertainty = rolling_correlation_coefficient(full_labels_val[:,i], full_means_val[:,i], 5)
            else:
                raise NotImplementedError

            # NOTE true uncertainty ranges from -1 (high) to +1 (low), so rescale it like UME will do
            calibration_target_ = np.abs(true_uncertainty - 1) / 2
            calibration_result  = calibration_utilities_deprecated.calibrate(calibration_features, calibration_target_, full_vars[:,i], "isotonic_regression")
            full_vars_calibrated[:,i] = calibration_result
        
        data_to_store[f"subjecivities_pred_cal_on_{calibration_target}"] = full_vars_calibrated
            
        sbUMEs_cal, pebUMEs_cal, Cvs_cal = calculate_uncertainty_metrics(params, full_labels, full_means, full_vars_calibrated, full_subjectivities, method + " (cal.)", test_loader.dataset.partition, params.uncertainty_approach != None)
        print(f"CALIBRATED on {calibration_target}:\nsbUMEs: {sbUMEs_cal}\npebUMES{pebUMEs_cal}\nCvs: {Cvs_cal}\n")
    
    # NOTE benchmarking
    sbUMEs_cal, pebUMEs_cal, Cvs_cal = calculate_uncertainty_metrics(params, full_labels, full_means, None, full_subjectivities, "Benchmark", test_loader.dataset.partition, False, True)
    print(f"BENCHMARK:\nsbUMEs: {sbUMEs_cal}\npebUMES{pebUMEs_cal}\nCvs: {Cvs_cal}\n")

    uncertainty_utilities_global.save_uncertainties_to_file(f"local_uncertainties_{params.uncertainty_approach}_{'_'.join(params.emo_dim_set)}_{test_loader.dataset.partition}", data_to_store)