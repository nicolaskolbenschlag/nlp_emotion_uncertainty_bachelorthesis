# Calibrated and Uncertainty-aware Multimodal Emotion Recognition

This repository contains the code for my bachelorthesis at University of Augsburg.

Imagine a neural network, which makes a prediction. But how confident (or the opposite uncertain) is the model about this prediction?

1. Define true uncertainty.
2. Measure/predict uncertainty, alongside with the actual prediction.
3. A model is referred to as *well-calibrated*, if true uncertainty matches predicted uncertainty.

Usually true uncertainty is defined as the prediction error, so we want the models uncertainty to be high if its error is high and vice-versa

## Types of Uncertainty

1. Aleatory (Data) Uncertainty:

* Caused by noise in the sample, as it leads to non-determination of an estimation problem.
* Imagine the landing point of an arrow.

2. Epistemic (Model/Knowledge) Uncertainty:

* Appears if the model lacks in knowledge on a sample.
* E.g., the image of a ship, fed into a cat-vs.-dog-classifier

## Measuring Predictive Uncertainty

The first step for getting appropriate measuremets of predictive uncertainty. Therefore we provide an extensive overview about recent approaches.

### Bayesian Modelling

1. Obtain multiple forecasts for yො (with possibly different outcomes).
2. Use the mean of them as final prediction and the variance among them as measurement for uncertainty.
3. Approaches:

* Monte Carlo Dropout: Enable dropout during inference.
* Ensemble Averaging: Train multiple model from varying starting point (seeds).

## Aim of this work

* Measure predictive uncertainty for the task of emotion recognition with MuSe-CaR.
* Investigate subjectivity among annotations as source of predictive uncertainty.

We apply two main conceptional approaches:

* **Local uncertainty** quantification: for now, anything happens **per time step**, or *locally*.
* **Global uncertainty** quantification: measurements are done for **multiple time steps** (sub-samples, respectively) together, or *globally*.

### Emotion Recognition

Prediction of the emotional state, here described by valence (sentiment) and arousal (excitement).

### MuSe 2021

The [MuSe CaR dataset](https://www.muse-challenge.org/) contains YouTube videos of car reviews contiguously (one label per time step) annotated by valence and arousal. For each sample, there are available 5 annotations by different human annotators. Further, there exists a *ground-truth* calculated, from the multiple annotations, by fusion techniques.

## Replacing Variance by Correlation

The annotation procedure, used for the data set of our experiments, the human annotators used joysticks (only up or down) to create the contiguous labelling over time. So when determining their annotation, they were actually forced to evaluate each time step relatively to the last one. More important than the actual value for the emotional state is the relative movement between two (or more) opinions on it, because time steps aren’t rated isolated.

## True Uncertainty

We adapted the idea of comparing true uncertainty and predicted/quantified uncertainty and thought about ways for properly quantification of true uncertainty for the task of continuous emotion recognition.

### Predictive Performance

First, we defined true uncertainty as the **prediction error**. The prediction error is *where we **want** uncertainty to appear*. This intuition is usually used in related papers. It is crucial in any application, because you need to know, if you can trust your model's confidence.

### Subjectivity among Raters

Of course, there can be multiple factors that cause predictive uncertainty, but we believe **disagreement among annotators** should be a main driver, respectively it is *where we would **expect** uncertainty to come from*. So it can be used as reference object to evaluate the feasibility of a method for quantifiying predictive uncertainty and as an indicator to learn more about the sources of uncertainty in this particular prediction task.

![uncalibrated](images/MC_Dropout_UNCALIBRATED_(700-800).jpg)

This plot shows the model's prediction (blue) and the quantified uncertainty (lightblue). As larger the lightblue area, as less confident was the model in its prediction. In this particular situation we used [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to make the model quantifying its confidence.

Further, the yellow line represents the true, or at least expected, uncertainty, which is defined as the average pearson correlation coefficient between multiple annotations. A correlation of +1 means that all annotators did perfectly agree (zero subjectivity/total objectivity) about the sample's annotation, as while -1 means negative correlation, so they absolutely disagreed (total subjectivity/zero objectivity).

What we expect (or at least hope) to observe is that, the model's uncertainty correlates with the subjectivity of the annotation. This would mean that we observe larger lightblue areas for smaller yellow values.

## Running the Code

`MuSe-LSTM-Attention-baseline-model\emotion_recognition\config.py` needs to be configured according to local files.

Requirements: `pip install -r requirements.txt`

### Local Uncertainty

Warning: deprecated! The experiments on *local uncertainty* do not work, or have negative results, respectively. Please refer to the thesis for details on that and on the experiments on *global* uncertainty measurement for positive results.

#### Monte Carlo Dropout (local)

Valence:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5`

Arousal:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5`

#### tCCC

Valence:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC`

Arousal:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC`

#### Staight Subjectivity

Valence & loss CCC:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc`

Valence & loss MSE:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse`

Arousal & loss CCC:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc --load_subjectivity_from_file`

Arousal & loss MSE:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse --load_subjectivity_from_file`

### Global Uncertainty

These commands store the measured uncertainty, for **later investigation** to files (`MuSe-LSTM-Attention-baseline-model/output/data/saved_uncertainties`). Use `extensive_uncertainty_statistics.py` to obtain metrics or `reliability_diagram.py` for plots.

#### Ensemble Averaging

Valence:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty_ensemble.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 5 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach ensemble_averaging --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20`

Arousal:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty_ensemble.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 5 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach ensemble_averaging --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20`

#### Monte Carlo Dropout (global)

Valence:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20`

Arousal:

`python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set vggish --emo_dim_set arousal --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20`
