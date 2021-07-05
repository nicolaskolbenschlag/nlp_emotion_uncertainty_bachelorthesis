import extensive_uncertainty_statistics
import scipy.stats

if __name__ == "__main__":
    base_dir = "C:/Users/Nicolas Kolbenschlag/Documents/Studium/6. Semester SS21/Bachelorarbeit/Sicherungen/uncertainties"

    files = ["global_uncertainties_ensemble_averaging_20_valence_devel", "global_uncertainties_ensemble_averaging_20_valence_test"]
    files += ["global_uncertainties_monte_carlo_dropout_20_valence_devel", "global_uncertainties_monte_carlo_dropout_20_valence_test"]

    files += ["global_uncertainties_ensemble_averaging_20_arousal_devel", "global_uncertainties_ensemble_averaging_20_arousal_test"]
    files += ["global_uncertainties_monte_carlo_dropout_20_arousal_devel", "global_uncertainties_monte_carlo_dropout_20_arousal_test"]

    for filename in files:
        uncertainties = extensive_uncertainty_statistics.load_from_file(f"{base_dir}/{filename}.pkl")[0]        
        predictive_performance, subjectivity = uncertainties["prediction_scores"], uncertainties["subjectivities_global"]

        ccc = extensive_uncertainty_statistics.ccc(predictive_performance, subjectivity)
        pcc = scipy.stats.pearsonr(predictive_performance, subjectivity)[0]
        # print(pcc)
        # exit()
        print(f"{filename} \t CCC {round(ccc, 4)} PCC {round(pcc, 4)}")