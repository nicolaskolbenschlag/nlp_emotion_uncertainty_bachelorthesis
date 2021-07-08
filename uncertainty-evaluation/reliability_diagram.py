import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle

def calc_ence(real_uncertainty: np.array, predicted_uncertainty: np.ndarray, bins: int) -> float:
    real_uncertainty = np.abs(real_uncertainty - 1) / 2
    predicted_uncertainty = np.abs(predicted_uncertainty - 1) / 2

    ence = 0.
    norm = 0
    # bin_indicies = np.digitize(predicted_uncertainty, np.linspace(predicted_uncertainty.min(), predicted_uncertainty.max(), bins))
    bin_indicies = np.digitize(predicted_uncertainty, np.linspace(-1, 1, bins))

    for j in range(1, bins + 1):
        mask = bin_indicies == j
        
        if not mask.sum() > 0:
            continue
        norm += 1

        rmse = np.sqrt(np.mean(real_uncertainty[mask]))
        rmv = np.sqrt(np.mean(predicted_uncertainty[mask]))

        ence_ = np.abs(rmv - rmse)
        if rmv != 0.:
            ence_ /= rmv

        ence += ence_

    ence /= norm
    return ence

def calc_bins(confidence_true: np.array, confidence_pred: np.array, n_bins: int):
    bins = np.linspace(-1, 1, n_bins + 2)
    binned = np.digitize(confidence_pred, bins)
    out_preds, out_trues = [], []
    bins_counts = []
    for j in range(1, n_bins + 2):
        mask = binned == j
        if mask.sum() == 0:
            print(f"No values for bin {j}")
            continue
        
        out_trues += [confidence_true[mask].mean()]
        out_preds += [confidence_pred[mask].mean()]

        bins_counts += [mask.sum()]

    return out_preds, out_trues, bins_counts

def plot_diagram(confidence_true: np.array, confidence_pred: np.array, title: str, ax: matplotlib.axes.Axes, n_bins: int = 3, label_true_confidence: str = None, main_color: str = "blue", normalize_to_correlation_space: bool = False) -> None:
    
    def normalize_to_correlation_space_fn(array: np.array):
        out = array - array.min()
        out /= out.max()
        out = out * 2 - 1
        assert out.shape == array.shape and out.min() == -1 and out.max() == 1
        return out
    
    if normalize_to_correlation_space:
        confidence_true = normalize_to_correlation_space_fn(confidence_true)
        confidence_pred = normalize_to_correlation_space_fn(confidence_pred)
    
    ence, var = calc_ence(confidence_true, confidence_pred, n_bins), np.var(confidence_pred)
    ence_optimal_fit = calc_ence(confidence_pred, confidence_pred, n_bins)

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    # ax.set_xlabel("Predicted Confidence", fontsize=8)
    # ax.set_ylabel("True Confidence" if label_true_confidence is None else label_true_confidence, fontsize=8)

    ax.set_axisbelow(True)
    ax.grid(color="gray", linestyle="dashed")
    ax.plot([-1,1], [-1,1], "--", color="orange", linewidth=2)

    bins_preds, _, _ = calc_bins(np.linspace(-1, 1, n_bins + 2), np.linspace(-1, 1, n_bins + 2), n_bins)
    # ax.bar(bins_preds, [b+1 for b in bins_preds], bottom=-1, width=.15, alpha=.2, edgecolor="black", color="orange", hatch="\\")

    bins_preds, bins_trues, bins_counts = calc_bins(confidence_true, confidence_pred, n_bins - 1)
    for bin_pred, bin_true, count in zip(bins_preds, bins_trues, bins_counts):
        alpha = .7 * count / max(bins_counts) + .3
        ax.bar(bin_pred, bin_true + 1, bottom=-1, width=.15, edgecolor="black", color=main_color, alpha=alpha, zorder=3)

    sigma = r"$\sigma^2$"
    ECE_patch = matplotlib.patches.Patch(color=main_color, label=f"ENCE = {round(ence, 2)} {sigma} = {round(var, 4)}")
    ECE_optimal_fit_patch = matplotlib.patches.Patch(color="orange", label=f"ENCE = {round(ence_optimal_fit, 2)} (perfect calibration)")
    ax.legend(handles=[ECE_patch, ECE_optimal_fit_patch], prop={"size": 12})

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.locator_params(nbins=4)

    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")

def plot_all_uncertainties(stored_uncertainties: str, n_bins: int = 5, normalize: bool = False) -> None:
    with open(stored_uncertainties.format("test"), "rb") as file:
        data_test = pickle.load(file)
    
    with open(stored_uncertainties.format("devel"), "rb") as file:
        data_val = pickle.load(file)

    for emo_dim_, data_test in data_test.items():
        data_val = data_val[emo_dim_]

        # NOTE load uncertainties from files
        subjectivities_true_test = data_test["subjectivities_global"]
        subjectivities_true_val = data_val["subjectivities_global"]
        
        prediction_scores_test = data_test["prediction_scores"]
        prediction_scores_val = data_val["prediction_scores"]

        subjectivities_pred_uncal_test = data_test["subjectivities_pred_uncalibrated"]
        subjectivities_pred_uncal_val = data_val["subjectivities_pred_uncalibrated"]

        subjectivities_pred_cal_on_subjectivity_isotonic_test = data_test["subjectivities_pred_calibrated_on_subjectivity"]["isotonic_regression"]
        subjectivities_pred_cal_on_subjectivity_isotonic_val = data_val["subjectivities_pred_calibrated_on_subjectivity"]["isotonic_regression"]
        
        subjectivities_pred_cal_on_subjectivity_std_scaling_test = data_test["subjectivities_pred_calibrated_on_subjectivity"]["std_scaling"]
        subjectivities_pred_cal_on_subjectivity_std_scaling_val = data_val["subjectivities_pred_calibrated_on_subjectivity"]["std_scaling"]

        subjectivities_pred_cal_on_prediction_score_isotonic_test = data_test["subjectivities_pred_calibrated_on_prediction_score"]["isotonic_regression"]        
        subjectivities_pred_cal_on_prediction_score_isotonic_val = data_val["subjectivities_pred_calibrated_on_prediction_score"]["isotonic_regression"]
        
        subjectivities_pred_cal_on_prediction_score_std_scaling_test = data_test["subjectivities_pred_calibrated_on_prediction_score"]["std_scaling"]
        subjectivities_pred_cal_on_prediction_score_std_scaling_val = data_val["subjectivities_pred_calibrated_on_prediction_score"]["std_scaling"]
        
        # NOTE subjectivity
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        # fig.suptitle(f"Subjectivity among Raters: {emo_dim} - Monte Carlo Dropout", fontsize=12)
        
        plot_diagram(subjectivities_true_val, subjectivities_pred_uncal_val, title="Uncalibrated [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[0,0], main_color="red", normalize_to_correlation_space=normalize)
        plot_diagram(subjectivities_true_test, subjectivities_pred_uncal_test, title="Uncalibrated [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[0,1], main_color="red", normalize_to_correlation_space=normalize)

        plot_diagram(subjectivities_true_val, subjectivities_pred_cal_on_subjectivity_isotonic_val, title="Isotonic Regression [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[1,0], main_color="red", normalize_to_correlation_space=normalize)
        plot_diagram(subjectivities_true_test, subjectivities_pred_cal_on_subjectivity_isotonic_test, title="Isotonic Regression [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[1,1], main_color="red", normalize_to_correlation_space=normalize)

        # plot_diagram(subjectivities_true_val, subjectivities_pred_cal_on_subjectivity_std_scaling_val, title="Std. Scaling [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[2,0], main_color="red", normalize_to_correlation_space=normalize)
        # plot_diagram(subjectivities_true_test, subjectivities_pred_cal_on_subjectivity_std_scaling_test, title="Std. Scaling [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[2,1], main_color="red", normalize_to_correlation_space=normalize)

        plt.show()
        
        # NOTE prediction score
        fig, axs = plt.subplots(2, 2, figsize=(10,10))
        # fig.suptitle(f"Subjectivity among Raters: {emo_dim} - Monte Carlo Dropout", fontsize=12)

        plot_diagram(prediction_scores_val, subjectivities_pred_uncal_val, title="Uncalibrated [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[0,0], normalize_to_correlation_space=normalize)
        plot_diagram(prediction_scores_test, subjectivities_pred_uncal_test, title="Uncalibrated [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[0,1], normalize_to_correlation_space=normalize)
        
        plot_diagram(prediction_scores_val, subjectivities_pred_cal_on_prediction_score_isotonic_val, title="Isotonic Regression [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[1,0], normalize_to_correlation_space=normalize)        
        plot_diagram(prediction_scores_test, subjectivities_pred_cal_on_prediction_score_isotonic_test, title="Isotonic Regression [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[1,1], normalize_to_correlation_space=normalize)

        # plot_diagram(prediction_scores_val, subjectivities_pred_cal_on_prediction_score_std_scaling_val, title="Std. Scaling [devel]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[2,0], normalize_to_correlation_space=normalize)
        # plot_diagram(prediction_scores_test, subjectivities_pred_cal_on_prediction_score_std_scaling_test, title="Std. Scaling [test]", label_true_confidence="Subjectivity Raters", n_bins=n_bins, ax=axs[2,1], normalize_to_correlation_space=normalize)

        plt.show()

if __name__ == "__main__":
    base_dir = "C:/Users/Nicolas Kolbenschlag/Documents/Studium/6. Semester SS21/Bachelorarbeit/Sicherungen/uncertainties"
    
    # filename = base_dir + "/global_uncertainties_ensemble_averaging_20_arousal_{}.pkl"
    # filename = base_dir + "/global_uncertainties_ensemble_averaging_20_valence_{}.pkl"

    # filename = base_dir + "/global_uncertainties_monte_carlo_dropout_20_arousal_{}.pkl"
    # filename = base_dir + "/global_uncertainties_monte_carlo_dropout_20_valence_{}.pkl"

    filename = base_dir + "/global_uncertainties_ensemble_averaging_10_arousal_{}.pkl"
    
    plot_all_uncertainties(filename, normalize=False)