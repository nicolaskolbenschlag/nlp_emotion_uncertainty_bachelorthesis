# Calibrated and Uncertainty-aware Multimodal Emotion Recognition

This repository contains the code for my bachelorthesis at University of Augsburg.

## MuSe 2021

The [dataset](https://www.muse-challenge.org/) used in this project.

## Quantification of predictive uncertainty

The first step for getting appropriate measuremets of predictive uncertainty. Therefore we provide a extensive overview about recent approaches.

## Measurement of predictive uncertainty

Additionally we developed an approach for confirming the feasibility of quantifying predictive uncertainty by such a method.

## Example

### Uncalibrated confidence

This plot shows the model's prediction (blue) and the quantified uncertainty (lightblue). As larger the lightblue area, as less confident was the model in its prediction. In this particular situation we used [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to make the model quantifying its confidence.

Further, the yellow line represents the true, or at least expected, uncertainty, which is defined as the average pearson correlation coefficient between multiple annotations. A correlation of +1 means that all annotators did perfectly agree (zero subjectivity/total objectivity) about the sample's annotation, as while -1 means negative correlation, so they absolutely disagreed (total subjectivity/zero objectivity).

![uncalibrated](images/MC_Dropout_UNCALIBRATED_(700-800).jpg)

What we expect (or at least hope) to observe is that, the model's uncertainty correlates with the subjectivity of the annotation. This would mean that we observe larger lightblue areas for smalles yellow values.

We also developed a metric to measure this similarity. Of course can be multiple factors that cause predictive uncertainty, but we believe disagreement among annotators, which is noise at the end, should be a main driver. So it can be used as reference object to evaluate the feasibility of a method for quantifiying predictive uncertainty.

### Calibrated confidence

We used a simple calibration method to optimize the model's quantification of uncertainty. Therefore, we just scale it with a single scalar value, which was found by optimizing it on the validation set.

![calibrated](images/MC_Dropout_CALIBRATED_(700-800).jpg)
