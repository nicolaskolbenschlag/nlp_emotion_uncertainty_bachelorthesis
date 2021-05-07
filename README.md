# Calibrated and Uncertainty-aware Multimodal Emotion Recognition

This repository contains the code for my bachelorthesis at University of Augsburg.

## MuSe 2021

We used the [MuSe CaR dataset](https://www.muse-challenge.org/) in this project. It contains multi-model features of car-review videos annotated by their emotional state. The circumstance that each sample was annotated by 5 different annotators, which are all available, and the fact that emotion recongintion is a highly subjective task, makes this dataset very well suited for investigation of uncertainty methods.

## Quantification of predictive uncertainty

The first step for getting appropriate measuremets of predictive uncertainty. Therefore we provide an extensive overview about recent approaches.

## The Uncertainty Measurement Error (UME)

Additionally we developed an approach for confirming the feasibility of quantifying predictive uncertainty by such a method.

We developed a metric called UME (Uncertainty Measurement Error) to measure this similarity. It is based on the ENCE proposed by Levi et al. in [this paper](https://arxiv.org/abs/1905.11659). The authors use RMSE as measurement for true uncertainty in non-timeseries-target regression tasks. But the RMSE is not a suitable metric for our data. Mostly, corrleations measured over a certain period of time is a much more suitable performance metric for time series prediction tasks, than just the distance at each timestep considered separately.

### Multiple understandings of uncertainty

Therefore, we adapted the idea of comparing true uncertainty and predicted/quantified uncertainty and thought about ways for properly quantification of true uncertainty for the task of continuous emotion recognition.

#### The subjectivity-based UME (sbUME)

Of course, there can be multiple factors that cause predictive uncertainty, but we believe **disagreement among annotators** should be a main driver, respectively it is *where we would **expect** uncertainty to come from*. So it can be used as reference object to evaluate the feasibility of a method for quantifiying predictive uncertainty and as an indicator to learn more about the sources of uncertainty in this particular prediction task.

![uncalibrated](images/MC_Dropout_UNCALIBRATED_(700-800).jpg)

This plot shows the model's prediction (blue) and the quantified uncertainty (lightblue). As larger the lightblue area, as less confident was the model in its prediction. In this particular situation we used [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to make the model quantifying its confidence.

Further, the yellow line represents the true, or at least expected, uncertainty, which is defined as the average pearson correlation coefficient between multiple annotations. A correlation of +1 means that all annotators did perfectly agree (zero subjectivity/total objectivity) about the sample's annotation, as while -1 means negative correlation, so they absolutely disagreed (total subjectivity/zero objectivity).

What we expect (or at least hope) to observe is that, the model's uncertainty correlates with the subjectivity of the annotation. This would mean that we observe larger lightblue areas for smaller yellow values.

#### The prediction-error-based UME (pebUME)

We can also calculate to UME between the quantified/predicted uncertianty and the **prediction error** (which than acts as measurement for true uncertainty). This might be useful, because the prediction error is *where we **want** uncertainty to appear*. This intuition is usually used in related papers. It is crucial in any application, because you need to know, if you can trust your model's confidence.

#### Results

<table>
    <tr>
        <th colspan="4">Randomly quantified uncertainty</th>
    </tr>
    <tr>
        <th></th>
        <th colspan="4">Uncalibrated</th>
        <th colspan="4">Calibrated with Isotonic Regression</th>
    </tr>
    <tr>
        <th>Seed</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
        <th>Cv</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
        <th>Cv</th>
    </tr>
    <tr>
        <td>314</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>315</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>&#8709;</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="9">Monte Carlo Dropout</th>
    </tr>
    <tr>
        <th></th>
        <th colspan="4">Uncalibrated</th>
        <th colspan="4">Calibrated with Isotonic Regression</th>
    </tr>
    <tr>
        <th>Seed</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
        <th>Cv</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
        <th>Cv</th>
    </tr>
    <tr>
        <td>314</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>315</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>&#8709;</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>

#### Residuals

The exact definition and computation of true uncertainty heavily depends on the actual data and its natrue, so this implementation can't be generalized from the current stand immediately to other datasets and prediction tasks. But the basic framework can be transferred.
