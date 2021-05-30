import numpy as np
import torch

import uncertainty_utilities
import calibration_utilities_deprecated

def mean_absolute_error(x: np.array, y: np.array) -> float:
    return (x - y).abs().mean()

def outputs_mc_dropout_global(model, test_loader, params, n_ensemble_members = 5):
    model.train()
    full_means, full_subjectivities_pred, full_labels, full_subjectivities_global = [], [], [], []
    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities, subjectivities_global = batch_data
            if params.gpu is not None:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels.cuda()
            preds = [model(features, feature_lens).cpu().detach().squeeze(0).numpy() for _ in range(n_ensemble_members)]
            means = np.mean(preds, axis=0)
            
            labels = labels.cpu().detach().squeeze(0).numpy()
            subjectivities_global = subjectivities_global.squeeze(0).numpy()

            subjectivities_pred = []
            for _ in range(means.shape[1]):
                subj_dim = []
                
                for i, pred_1 in enumerate(preds):
                    for pred_2 in preds[i+1:]:
                        ccc = uncertainty_utilities.ccc_score(pred_1, pred_2)
                        subj_dim += [ccc]
                
                subj_dim = np.mean(subj_dim)
                subjectivities_pred += [subj_dim]
            
            full_means += [means]
            full_subjectivities_pred += [subjectivities_pred]
            full_labels += [labels]
            full_subjectivities_global += [subjectivities_global]
    
    return full_means, full_subjectivities_pred, full_labels, full_subjectivities_global

def evaluate_uncertainty_measurement_global(params, model, test_loader, val_loader):
    print("-" * 20 + "TEST" + "-" * 20)
    evaluate_uncertainty_measurement_global_help(params, model, test_loader, val_loader)
    print("-" * 20 + "DEVEL" + "-" * 20)
    evaluate_uncertainty_measurement_global_help(params, model, val_loader, val_loader)

def calculate_metrics(subjectivities_pred, subjectivities_global, prediction_scores):
    # NOTE calculate global equivalent to sbUME: Global sbUME (GsbUME)
    GsbUME = mean_absolute_error(subjectivities_global, subjectivities_pred)
    
    # NOTE compare to GsbUME with guessed (normally distributed) subjectivities
    subjectivities_pred_rand = np.random.normal(subjectivities_global.mean(), subjectivities_global.std(), subjectivities_pred.shape)
    GsbUME_rand = mean_absolute_error(subjectivities_global, subjectivities_pred_rand)

    # NOTE calculate global equivalent to pebUME: Global pebUME (GpebUME)
    GpebUME = mean_absolute_error(prediction_scores, subjectivities_pred)

    # NOTE compare to GpebUME with guessed (normally distributed) prediction errors
    subjectivities_pred_rand = np.random.normal(prediction_scores.mean(), prediction_scores.std(), subjectivities_pred.shape)
    GpebUME_rand = mean_absolute_error(prediction_scores, subjectivities_pred_rand)

    # NOTE measure similarity between true subjectivity among annotations and prediction error
    vs = mean_absolute_error(prediction_scores, subjectivities_global)

    return GsbUME, GsbUME_rand, GpebUME, GpebUME_rand, vs

def evaluate_uncertainty_measurement_global_help(params, model, test_loader, val_loader):
    if params.uncertainty_approach == "monte_carlo_dropout":
        prediction_fn = outputs_mc_dropout_global
    
    else:
        raise NotImplementedError
    
    # NOTE full_subjectivities_pred represents the model's uncertainty, confidence respectively
    full_means, full_subjectivities_pred, full_labels, full_subjectivities_global = prediction_fn(model, test_loader, params)
    full_means_val, full_subjectivities_pred_val, full_labels_val, full_subjectivities_global_val = prediction_fn(model, val_loader, params)

    GsbUMEs, GsbUME_rands, GpebUMEs, GpebUME_rands, prediction_error_vs_subjectivity = [], [], [], [], []
    GsbUMEs_cal_subj, GpebUMEs_cal_subj = [], []
    GsbUMEs_cal_err, GpebUMEs_cal_err = [], []
    for emo_dim in range(full_means[0].shape[1]):

        subjectivities_pred = np.array(full_subjectivities_pred)[:,emo_dim]
        subjectivities_global = np.array(full_subjectivities_global)[:,emo_dim]
        prediction_scores = np.array([uncertainty_utilities.ccc_score(full_means[i][:,emo_dim], full_labels[i][:,emo_dim]) for i in range(len(full_means))])
        assert subjectivities_pred.shape == subjectivities_global.shape == prediction_scores.shape

        # NOTE uncalibrated measurements
        GsbUME, GsbUME_rand, GpebUME, GpebUME_rand, vs = calculate_metrics(subjectivities_pred, subjectivities_global, prediction_scores)
        GsbUMEs += [GsbUME]; GsbUME_rands += [GsbUME_rand]; GpebUMEs += [GpebUME]; GpebUME_rands += [GpebUME_rand]; prediction_error_vs_subjectivity += [vs]

        # NOTE re-calibration
        calibration_features_train = np.array(full_subjectivities_pred_val)[:,emo_dim]
        calibration_features = subjectivities_pred

        # NOTE calibration target: subjectivity among annotations
        calibration_target_train = np.array(full_subjectivities_global_val)[:,emo_dim]
        calibration_target_pred = calibration_utilities_deprecated.calibrate(calibration_features_train, calibration_target_train, calibration_features, "isotonic_regression")
        # NOTE only obtain metrics that are affected by calibration
        GsbUME_cal_subj, _, GpebUME_cal_subj, _, _ = calculate_metrics(calibration_target_pred, subjectivities_global, prediction_scores)
        GsbUMEs_cal_subj += [GsbUME_cal_subj]; GpebUMEs_cal_subj += [GpebUME_cal_subj]

        # NOTE calibration target: prediction error
        calibration_target_train = np.array([uncertainty_utilities.ccc_score(full_means_val[i][:,emo_dim], full_labels_val[i][:,emo_dim]) for i in range(len(full_means_val))])
        calibration_target_pred = calibration_utilities_deprecated.calibrate(calibration_features_train, calibration_target_train, calibration_features, "isotonic_regression")
        GsbUME_cal_err, _, GpebUME_cal_err, _, _ = calculate_metrics(calibration_target_pred, subjectivities_global, prediction_scores)
        GsbUMEs_cal_err += [GsbUME_cal_err]; GpebUMEs_cal_err += [GpebUME_cal_err]
    
    print("Uncalibrated scores and benchmarking with random uncertainty quntification:")
    print(f"GsbUME: {GsbUMEs}, rand. GsbUME: {GsbUME_rands}")
    print(f"GpebUME: {GpebUMEs}, rand. GpebUME: {GpebUME_rands}")
    print(f"true-subjectivity-vs.-prediction-error: {prediction_error_vs_subjectivity}")
    
    print("Calibrated on true subjectivity:")
    print(f"GsbUME: {GsbUMEs_cal_subj}")
    print(f"GpebUME: {GpebUMEs_cal_subj}")

    print("Calibrated on prediction score:")
    print(f"GsbUME: {GsbUMEs_cal_err}")
    print(f"GpebUME: {GpebUMEs_cal_err}")