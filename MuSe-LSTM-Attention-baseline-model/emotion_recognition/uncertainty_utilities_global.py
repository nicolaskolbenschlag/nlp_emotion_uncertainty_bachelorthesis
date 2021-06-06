import numpy as np
import torch
import sklearn.isotonic

import uncertainty_utilities

def mean_absolute_error(x: np.array, y: np.array) -> float:
    return np.abs(x - y).mean()

def multiple_preds_to_predicted_subjectivity(params, preds, means):
    subjectivities_pred = []
    
    for dim in range(means.shape[1]):
        subj_dim = []
        
        for i, pred_1 in enumerate(preds):
            for pred_2 in preds[i+1:]:
                
                if params.global_uncertainty_window is None:
                    ccc = uncertainty_utilities.ccc_score(pred_1[:,dim], pred_2[:,dim])
                    subj_dim += [ccc]
                
                else:
                    window = params.global_uncertainty_window
                    tmp = []
                    for i in range(0, len(pred_1[:,dim]) + 1 - window, window):
                        if len(pred_1[:,dim][i : i + window]) < window:
                            continue
                        tmp += [uncertainty_utilities.ccc_score(pred_1[:,dim][i : i + window], pred_2[:,dim][i : i + window])]
                    subj_dim += [tmp]
        
        if params.global_uncertainty_window is None:
            subj_dim = np.mean(subj_dim)
        else:
            subj_dim = np.mean(subj_dim, axis=0)
        
        subjectivities_pred += [subj_dim]
    
    return subjectivities_pred

def calculate_prediction_scores(full_means, full_labels, emo_dim, params):
    if params.global_uncertainty_window is None:
        return np.array([uncertainty_utilities.ccc_score(full_means[i][:,emo_dim], full_labels[i][:,emo_dim]) for i in range(len(full_means))])
    
    else:
        window = params.global_uncertainty_window
        out = []
        for idx, sample in enumerate(full_means):
            for i in range(0, len(sample[:,emo_dim]) + 1 - window, window):
                if len(sample[:,emo_dim][i : i + window]) < window:
                    continue
                ccc = uncertainty_utilities.ccc_score(sample[:,emo_dim][i : i + window], full_labels[idx][:,emo_dim][i : i + window])
                out += [ccc]
        return np.array(out)

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

            subjectivities_pred = multiple_preds_to_predicted_subjectivity(params, preds, means)
            
            full_means += [means]
            full_subjectivities_pred += [subjectivities_pred]
            full_labels += [labels]
            full_subjectivities_global += [subjectivities_global]
    
    return full_means, full_subjectivities_pred, full_labels, full_subjectivities_global

def outputs_ensemble_averaging_global(ensemble, test_loader, params):
    full_means, full_subjectivities_pred, full_labels, full_subjectivities_global = [], [], [], []

    with torch.no_grad():
        for _, batch_data in enumerate(test_loader, 1):
            features, feature_lens, labels, meta, subjectivities, subjectivities_global = batch_data

            labels = labels.cpu().detach().squeeze(0).numpy()
            full_labels += [labels]
            subjectivities_global = subjectivities_global.squeeze(0).numpy()
            full_subjectivities_global += [subjectivities_global]

            preds = []
            for model in ensemble:
                model.eval()

                if params.gpu is not None:
                    model.cuda()
                    features = features.cuda()
                    feature_lens = feature_lens.cuda()
                    labels = labels.cuda()
                
                pred = model(features, feature_lens).cpu().detach().squeeze(0).numpy()
                preds += [pred]
            
            means = np.mean(preds, axis=0)
            full_means += [means]

            subjectivities_pred = multiple_preds_to_predicted_subjectivity(params, preds, means)
            
            full_subjectivities_pred += [subjectivities_pred]
    
    return full_means, full_subjectivities_pred, full_labels, full_subjectivities_global

def calculate_metrics(subjectivities_pred, subjectivities_global, prediction_scores, normalize: bool = False):

    assert subjectivities_pred.shape == subjectivities_global.shape == prediction_scores.shape, "should be: {subjectivities_pred.shape} == {subjectivities_global.shape} == {prediction_scores.shape}"

    if normalize:
        subjectivities_pred -= subjectivities_pred.min()
        subjectivities_pred /= subjectivities_pred.max()

        subjectivities_global -= subjectivities_global.min()
        subjectivities_global /= subjectivities_global.max()

        prediction_scores -= prediction_scores.min()
        prediction_scores /= prediction_scores.max()

    GsbUME, GsbUME_rand, GpebUME, GpebUME_rand, vs = {}, {}, {}, {}, {}

    metric_fns = [mean_absolute_error, uncertainty_utilities.ccc_score]
    for i, metric in enumerate(["mae", "ccc"]):
        metric_fn = metric_fns[i]

        # NOTE calculate global equivalent to sbUME: Global sbUME (GsbUME)
        GsbUME[metric] = metric_fn(subjectivities_global, subjectivities_pred)
        
        # NOTE compare to GsbUME with guessed (normally distributed) subjectivities
        subjectivities_pred_rand = np.random.normal(subjectivities_global.mean(), subjectivities_global.std(), subjectivities_pred.shape)
        GsbUME_rand[metric] = metric_fn(subjectivities_global, subjectivities_pred_rand)

        # NOTE calculate global equivalent to pebUME: Global pebUME (GpebUME)
        GpebUME[metric] = metric_fn(prediction_scores, subjectivities_pred)

        # NOTE compare to GpebUME with guessed (normally distributed) prediction errors
        subjectivities_pred_rand = np.random.normal(prediction_scores.mean(), prediction_scores.std(), subjectivities_pred.shape)
        GpebUME_rand[metric] = metric_fn(prediction_scores, subjectivities_pred_rand)

        # NOTE measure similarity between true subjectivity among annotations and prediction error
        vs[metric] = metric_fn(prediction_scores, subjectivities_global)

    return GsbUME, GsbUME_rand, GpebUME, GpebUME_rand, vs

def calibrate(val_uncalibrated: np.array, val_calibrated: np.array, test_uncalibrated: np.array, method: str) -> np.array:
    if method == "isotonic_regression":
        calibrator = sklearn.isotonic.IsotonicRegression(out_of_bounds="clip", y_min=-1, y_max=1).fit(val_uncalibrated, val_calibrated)
        return calibrator.predict(test_uncalibrated)
    else:
        raise NotImplementedError

def evaluate_uncertainty_measurement_global_help(params, model, test_loader, val_loader):
    if params.uncertainty_approach == "monte_carlo_dropout":
        prediction_fn = outputs_mc_dropout_global
    
    elif params.uncertainty_approach == "ensemble_averaging":
        prediction_fn = outputs_ensemble_averaging_global
    
    else:
        raise NotImplementedError
    
    # NOTE full_subjectivities_pred represents the model's uncertainty, confidence respectively
    full_means, full_subjectivities_pred, full_labels, full_subjectivities_global = prediction_fn(model, test_loader, params)
    full_means_val, full_subjectivities_pred_val, full_labels_val, full_subjectivities_global_val = prediction_fn(model, val_loader, params)

    if not params.global_uncertainty_window is None:
        
        # NOTE reshape from (num_samples, dims, num_subsamples) to (num_samples * num_subsamples, dims)
        def flatten_subjectivities_of_subsamples(subjectivities, params):
            out = []
            for sample in subjectivities:
                assert len(sample) == len(params.emo_dim_set)

                for i_subsample in range(len(sample[0])):
                    new_sample = []
                    none_occurrence = False
                    
                    for i_dim in range(len(sample)):
                        subsample = sample[i_dim][i_subsample]
                        
                        if not torch.isnan(torch.tensor(subsample)):
                            if torch.is_tensor(subsample): subsample = subsample.item()
                            new_sample += [subsample]
                            assert not none_occurrence, "ensure that subsumple is padded with nones at the end and no nones elsewhere"
                        else:
                            none_occurrence = True
                    
                    # NOTE only if new_sample is not only nones for padding
                    if len(new_sample) == len(params.emo_dim_set):
                        out += [new_sample]

            out = np.array(out)
            print(out.shape)
            # print(out)
            assert out.shape[1:] == (len(params.emo_dim_set),)
            return out
            
        full_subjectivities_pred = flatten_subjectivities_of_subsamples(full_subjectivities_pred, params)
        full_subjectivities_global = flatten_subjectivities_of_subsamples(full_subjectivities_global, params)
        assert full_subjectivities_pred.shape == full_subjectivities_global.shape
        full_subjectivities_pred_val = flatten_subjectivities_of_subsamples(full_subjectivities_pred_val, params)
        full_subjectivities_global_val = flatten_subjectivities_of_subsamples(full_subjectivities_global_val, params)
        assert full_subjectivities_pred_val.shape == full_subjectivities_global_val.shape


    GsbUMEs, GsbUME_rands, GpebUMEs, GpebUME_rands, prediction_error_vs_subjectivity = [], [], [], [], []
    GsbUMEs_cal_subj, GpebUMEs_cal_subj = [], []
    GsbUMEs_cal_err, GpebUMEs_cal_err = [], []
    for emo_dim in range(full_means[0].shape[1]):

        subjectivities_pred = np.array(full_subjectivities_pred)[:,emo_dim]
        subjectivities_global = np.array(full_subjectivities_global)[:,emo_dim]
        
        # prediction_scores = np.array([uncertainty_utilities.ccc_score(full_means[i][:,emo_dim], full_labels[i][:,emo_dim]) for i in range(len(full_means))])
        prediction_scores = calculate_prediction_scores(full_means, full_labels, emo_dim, params)

        assert subjectivities_pred.shape == subjectivities_global.shape == prediction_scores.shape, "should be: {subjectivities_pred.shape} == {subjectivities_global.shape} == {prediction_scores.shape}"

        # NOTE uncalibrated measurements
        GsbUME, GsbUME_rand, GpebUME, GpebUME_rand, vs = calculate_metrics(subjectivities_pred, subjectivities_global, prediction_scores, params.normalize_uncalibrated_global_uncertainty_measurement)
        GsbUMEs += [GsbUME]; GsbUME_rands += [GsbUME_rand]; GpebUMEs += [GpebUME]; GpebUME_rands += [GpebUME_rand]; prediction_error_vs_subjectivity += [vs]

        # NOTE re-calibration
        calibration_features_train = np.array(full_subjectivities_pred_val)[:,emo_dim]
        calibration_features = subjectivities_pred

        # NOTE calibration target: subjectivity among annotations
        calibration_target_train = np.array(full_subjectivities_global_val)[:,emo_dim]

        # print(calibration_features_train)
        # print(calibration_target_train)
        # print(calibration_features)

        calibration_target_pred = calibrate(calibration_features_train, calibration_target_train, calibration_features, "isotonic_regression")
        # NOTE only obtain metrics that are affected by calibration
        GsbUME_cal_subj, _, GpebUME_cal_subj, _, _ = calculate_metrics(calibration_target_pred, subjectivities_global, prediction_scores, False)
        GsbUMEs_cal_subj += [GsbUME_cal_subj]; GpebUMEs_cal_subj += [GpebUME_cal_subj]

        # NOTE calibration target: prediction error
        
        # calibration_target_train = np.array([uncertainty_utilities.ccc_score(full_means_val[i][:,emo_dim], full_labels_val[i][:,emo_dim]) for i in range(len(full_means_val))])
        calibration_target_train = calculate_prediction_scores(full_means_val, full_labels_val, emo_dim, params)
        
        calibration_target_pred = calibrate(calibration_features_train, calibration_target_train, calibration_features, "isotonic_regression")
        GsbUME_cal_err, _, GpebUME_cal_err, _, _ = calculate_metrics(calibration_target_pred, subjectivities_global, prediction_scores, False)
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

def evaluate_uncertainty_measurement_global(params, model, test_loader, val_loader):
    print("-" * 20 + "TEST" + "-" * 20)
    evaluate_uncertainty_measurement_global_help(params, model, test_loader, val_loader)
    print("-" * 20 + "DEVEL" + "-" * 20)
    evaluate_uncertainty_measurement_global_help(params, model, val_loader, val_loader)