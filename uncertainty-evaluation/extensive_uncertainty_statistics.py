import numpy as np
import pickle
import sys

# NOTE -------------------- STATISTICS --------------------

def ence(real_uncertainty: np.array, predicted_uncertainty: np.ndarray, bins: int = 5) -> float:
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

def mae(x: np.array, y: np.array) -> float:
    return np.abs(x - y).mean()

def ccc(x: np.array, y: np.array) -> float:
    x_mean, y_mean = np.mean(x), np.mean(y)
    cov_mat = np.cov(x, y)
    covariance = cov_mat[0,1]
    x_var, y_var = cov_mat[0,0], cov_mat[1,1]
    ccc = 2. * covariance / (x_var + y_var + (x_mean - y_mean) ** 2)
    return ccc

def cv(y_pred_var: np.ndarray) -> float:
    mean_of_var = np.mean(y_pred_var)
    cv = np.sqrt(np.power(y_pred_var - mean_of_var, 2).sum() / (len(y_pred_var) - 1)) / mean_of_var
    return cv

STATS_COMPARE = [("ENCE", ence), ("MAE", mae), ("CCC", ccc)]
# STATS_VOLATILITY = [("Cv", cv), ("Var", np.var)]
STATS_VOLATILITY = [("Var", np.var)]

# NOTE -------------------- FILE HANDLING --------------------

def benchmark(array: np.array) -> np.array:
    return np.random.normal(array.mean(), array.std(), array.shape)

def load_from_file(filename: str) -> dict:
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data

def load_global_uncertainties(filename: str) -> dict:
    data = load_from_file(filename)
    assert len(data) == 1, "only support one concurrent emo. dim."
    data = data[0]

    # NOTE store uncertainties as tuples: (true, predicted, benchmark)
    uncertainties = {}

    subjectivities_true = data["subjectivities_global"]
    prediction_scores = data["prediction_scores"]
    subjectivities_pred_uncal = data["subjectivities_pred_uncalibrated"]
    subjectivities_pred_cal_on_subjectivity_isotonic = data["subjectivities_pred_calibrated_on_subjectivity"]["isotonic_regression"]
    subjectivities_pred_cal_on_subjectivity_std_scaling = data["subjectivities_pred_calibrated_on_subjectivity"]["std_scaling"]
    subjectivities_pred_cal_on_prediction_score_isotonic = data["subjectivities_pred_calibrated_on_prediction_score"]["isotonic_regression"]
    subjectivities_pred_cal_on_prediction_score_std_scaling = data["subjectivities_pred_calibrated_on_prediction_score"]["std_scaling"]

    benchmark_subjectivities = benchmark(subjectivities_true)
    # uncertainties["subjectivities_uncalibrated"] = (subjectivities_true, subjectivities_pred_uncal, benchmark_subjectivities)
    # uncertainties["subjectivities_calibrated_with_ir"] = (subjectivities_true, subjectivities_pred_cal_on_subjectivity_isotonic, benchmark_subjectivities)
    # uncertainties["subjectivities_calibrated_with_std"] = (subjectivities_true, subjectivities_pred_cal_on_subjectivity_std_scaling, benchmark_subjectivities)

    benchmark_prediction_scores = benchmark(prediction_scores)
    uncertainties["prediction_scores_uncalibrated"] = (prediction_scores, subjectivities_pred_uncal, benchmark_prediction_scores)
    uncertainties["prediction_scores_calibrated_with_ir"] = (prediction_scores, subjectivities_pred_cal_on_prediction_score_isotonic, benchmark_prediction_scores)
    uncertainties["prediction_scores_calibrated_with_std"] = (prediction_scores, subjectivities_pred_cal_on_prediction_score_std_scaling, benchmark_prediction_scores)

    return uncertainties

def load_local_uncertainties(filename: str) -> dict:
    data = load_from_file(filename)
    data = {key: value[:,0] for key, value in data.items()}

    uncertainties = {}

    subjectivities_true = data["subjectivities"]
    prediction_scores = data["rolling_prediction_scores_3"]
    subjectivities_pred_uncal = data["subjectivities_pred"]
    subjectivities_pred_cal_on_subjectivity_isotonic = data["subjecivities_pred_cal_on_subjectivity"]
    subjectivities_pred_cal_on_prediction_score_isotonic = data["subjecivities_pred_cal_on_rolling_error_3"]

    benchmark_subjectivities = benchmark(subjectivities_true)
    uncertainties["subjectivities_uncalibrated"] = (subjectivities_true, subjectivities_pred_uncal, benchmark_subjectivities)
    # uncertainties["subjectivities_calibrated_with_ir"] = (subjectivities_true, subjectivities_pred_cal_on_subjectivity_isotonic, benchmark_subjectivities)

    benchmark_prediction_scores = benchmark(prediction_scores)
    uncertainties["prediction_scores_uncalibrated"] = (prediction_scores, subjectivities_pred_uncal, benchmark_prediction_scores)
    # uncertainties["prediction_scores_calibrated_with_ir"] = (prediction_scores, subjectivities_pred_cal_on_prediction_score_isotonic, benchmark_prediction_scores)

    return uncertainties

def print_all_stats(true: np.array, pred: np.array, key: str):
    tabs = "\t" * ((50 - len(key)) // 5)
    msg = f"{key}{tabs}"
    for stat in STATS_COMPARE:
        score = stat[1](true, pred)
        msg += f"\t{round(score, 4)}"
    for stat in STATS_VOLATILITY:
        score_true = stat[1](true)
        score_true = round(score_true, 4)
        msg += "\t" + str(score_true)
        score_pred = stat[1](pred)
        score_pred = round(score_pred, 4)
        msg += "\t" + str(score_pred)
    print(msg)

def calculate_and_print_statistics(uncertainties: dict) -> None:
    msg = "\t" * 11
    for stat in STATS_COMPARE:
        msg += f"\t{stat[0]}"
    for stat in STATS_VOLATILITY:
        msg += f"\t{stat[0]} true"
        msg += f"\t{stat[0]} pred"
    print(msg)

    for key, value in uncertainties.items():
        true, pred, benchmark = value

        print_all_stats(true, pred, key)
        print_all_stats(true, benchmark, f"{key} [BM]")
        print()

if __name__ == "__main__":
    file = open("stats.txt", "w")
    # sys.stdout = file

    base_dir = "C:/Users/Nicolas Kolbenschlag/Documents/Studium/6. Semester SS21/Bachelorarbeit/Sicherungen/uncertainties"

    print("GLOBAL UNCERTAINTIES")
    files = ["global_uncertainties_ensemble_averaging_20_valence_devel", "global_uncertainties_ensemble_averaging_20_valence_test"]
    files = ["global_uncertainties_monte_carlo_dropout_20_valence_devel", "global_uncertainties_monte_carlo_dropout_20_valence_test"]

    files = ["global_uncertainties_ensemble_averaging_20_arousal_devel", "global_uncertainties_ensemble_averaging_20_arousal_test"]
    # files = ["global_uncertainties_monte_carlo_dropout_20_arousal_devel", "global_uncertainties_monte_carlo_dropout_20_arousal_test"]

    # files = ["global_uncertainties_ensemble_averaging_10_arousal_devel", "global_uncertainties_ensemble_averaging_10_arousal_test"]
    
    # files = ["global_uncertainties_monte_carlo_dropout_20_valence_fasttext_devel", "global_uncertainties_monte_carlo_dropout_20_valence_fasttext_test"]

    for filename in files:
        print(filename)
        uncertainties = load_global_uncertainties(f"{base_dir}/{filename}.pkl")
        calculate_and_print_statistics(uncertainties)


    # print("SHORT-TERM UNCERTAINTIES")
    # # files = ["local_uncertainties_quantile_regression_arousal_devel", "local_uncertainties_quantile_regression_arousal_test"]
    # files = ["local_uncertainties_monte_carlo_dropout_arousal_devel", "local_uncertainties_monte_carlo_dropout_arousal_test"]

    # for filename in files:
    #     print(filename)
    #     uncertainties = load_local_uncertainties(f"{base_dir}/{filename}.pkl")
    #     calculate_and_print_statistics(uncertainties)
