import extensive_uncertainty_statistics
import numpy as np
import matplotlib.pyplot as plt

# files_fasttext = ["global_uncertainties_monte_carlo_dropout_20_valence_fasttext_devel", "global_uncertainties_monte_carlo_dropout_20_valence_fasttext_test"]
# files_bert = ["global_uncertainties_monte_carlo_dropout_20_valence_devel", "global_uncertainties_monte_carlo_dropout_20_valence_test"]

files_fasttext = ["global_uncertainties_ensemble_averaging_20_valence_fasttext_devel", "global_uncertainties_ensemble_averaging_20_valence_fasttext_test"]
files_bert = ["global_uncertainties_ensemble_averaging_20_valence_devel", "global_uncertainties_ensemble_averaging_20_valence_test"]

base_dir = "C:/Users/Nicolas Kolbenschlag/Documents/Studium/6. Semester SS21/Bachelorarbeit/Sicherungen/uncertainties"

### DEVEL ###

uncertainties_fasttext = extensive_uncertainty_statistics.load_from_file(f"{base_dir}/{files_fasttext[0]}.pkl")[0]
uncertainties_bert = extensive_uncertainty_statistics.load_from_file(f"{base_dir}/{files_bert[0]}.pkl")[0]

uncertainties_fasttext_abs = np.abs(np.mean(uncertainties_fasttext["subjectivities_pred_uncalibrated"]) - 1) / 2
uncertainties_bert_abs = np.abs(np.mean(uncertainties_bert["subjectivities_pred_uncalibrated"]) - 1) / 2

ccc_subjectivity_fasttext = extensive_uncertainty_statistics.ccc(uncertainties_fasttext["subjectivities_pred_uncalibrated"], uncertainties_fasttext["subjectivities_global"])
ccc_subjectivity_bert = extensive_uncertainty_statistics.ccc(uncertainties_bert["subjectivities_pred_uncalibrated"], uncertainties_bert["subjectivities_global"])

plt.scatter(ccc_subjectivity_fasttext, uncertainties_fasttext_abs, label="fastText [devel]", marker=r"$fastText_{devel}$", c="green", alpha=1, s=5000)
plt.scatter(ccc_subjectivity_bert, uncertainties_bert_abs, label="BERT [devel]", marker=r"$BERT_{devel}$", c="blue", alpha=1, s=5000)

### TEST ###

uncertainties_fasttext = extensive_uncertainty_statistics.load_from_file(f"{base_dir}/{files_fasttext[1]}.pkl")[0]
uncertainties_bert = extensive_uncertainty_statistics.load_from_file(f"{base_dir}/{files_bert[1]}.pkl")[0]

uncertainties_fasttext_abs = np.abs(np.mean(uncertainties_fasttext["subjectivities_pred_uncalibrated"]) - 1) / 2
uncertainties_bert_abs = np.abs(np.mean(uncertainties_bert["subjectivities_pred_uncalibrated"]) - 1) / 2

ccc_subjectivity_fasttext = extensive_uncertainty_statistics.ccc(uncertainties_fasttext["subjectivities_pred_uncalibrated"], uncertainties_fasttext["subjectivities_global"])
ccc_subjectivity_bert = extensive_uncertainty_statistics.ccc(uncertainties_bert["subjectivities_pred_uncalibrated"], uncertainties_bert["subjectivities_global"])

plt.scatter(ccc_subjectivity_fasttext, uncertainties_fasttext_abs, label="fastText [test]", marker=r"$fastText_{test}$", c="green", s=5000)
plt.scatter(ccc_subjectivity_bert, uncertainties_bert_abs, label="BERT [test]", marker=r"$BERT_{test}$", c="blue", s=5000)


# plt.scatter(0, 1, label="Complete Epistemic Uncertainty", marker="x", c="b")
# plt.scatter(1, 1, label="Complete Aleatory Uncertainty", marker="x", c="r")

plt.xlabel("Raters' Subjectivity in Model's Uncertainty [CCC]", fontsize=18)
plt.ylabel("Overall Uncertainty", fontsize=18)

# plt.legend()
plt.show()