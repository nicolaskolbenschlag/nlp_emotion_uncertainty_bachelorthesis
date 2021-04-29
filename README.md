# Calibrated and Uncertainty-aware Multimodal Emotion Recognition

This repository contains the code for my bachelorthesis at University of Augsburg.

## MuSe 2021

The [dataset](https://www.muse-challenge.org/) used in this project.

## Quantification of predictive uncertainty

The first step for getting appropriate measuremets of predictive uncertainty. Therefore we provide a extensive overview about recent approaches.

## Measurement of predictive uncertainty

Additionally we developed an approach for confirming the feasibility of quantifying predictive uncertainty by such a method.

### The Uncertainty Measurement Error (UME)

This plot shows the model's prediction (blue) and the quantified uncertainty (lightblue). As larger the lightblue area, as less confident was the model in its prediction. In this particular situation we used [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) to make the model quantifying its confidence.

Further, the yellow line represents the true, or at least expected, uncertainty, which is defined as the average pearson correlation coefficient between multiple annotations. A correlation of +1 means that all annotators did perfectly agree (zero subjectivity/total objectivity) about the sample's annotation, as while -1 means negative correlation, so they absolutely disagreed (total subjectivity/zero objectivity).

![uncalibrated](images/MC_Dropout_UNCALIBRATED_(700-800).jpg)

What we expect (or at least hope) to observe is that, the model's uncertainty correlates with the subjectivity of the annotation. This would mean that we observe larger lightblue areas for smaller yellow values.

#### Multiple understandings of uncertainty

We also developed a metric called UME (Uncertainty Measurement Error) to measure this similarity. Of course, there can be multiple factors that cause predictive uncertainty, but we believe **disagreement among annotators** should be a main driver, respectively it is *where we would **expect** uncertainty to come from*. So it can be used as reference object to evaluate the feasibility of a method for quantifiying predictive uncertainty and as an indicator to learn more about the sources of uncertainty in this particular prediction task.

We can also calculate to UME between the quantified/predicted uncertianty and the **prediction error** (which then acts as measurement for true uncertainty). This might be useful, because the prediction error is *where we **want** uncertainty to appear*. This intuition is usually used in related papers. It is crucial in any application, because you need to know, if you can trust your model's confidence.

The exact definition and computation of true uncertainty heavily depends on the actual data and its natrue, so this implementation can't be generalized from the current stand immediately to other datasets and prediction tasks, but the basic framework can be transferred.

#### Results

<table>
    <tr>
        <th></th>
        <th colspan="3">Randomly quantified uncertainty</th>
        <th colspan="3">Monte Carlo Dropout</th>
    </tr>
    <tr>
        <th>Seed</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
        <th>CCC</th>
        <th>sbUME</th>
        <th>pebUME</th>
    </tr>
    <tr>
        <td>314</td>
        <td>0.5833</td>
        <td>0.6906</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>315</td>
        <td>0.5820</td>
        <td>0.6959</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>316</td>
        <td>0.5900</td>
        <td>0.7159</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>&#8709;</td>
        <td>0.5851</td>
        <td>0.7008</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
</table>
