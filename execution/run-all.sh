#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features="bert-4"
emo_dim="valence"

calibration_targets="subjectivity rolling_error_3"
ume_scaling_windows="None 10 200"

# MONTE CARLO DROPOUT
for t in $calibration_targets; do
    cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --calibration_target $t"
    python3 $cmd
    for w in $ume_scaling_windows; do
        python3 "$cmd --ume_rolling_scaling_window $w"
    done
done

# RANDOM UNCERTAINTY QUANTIFICATIONS
for w in $ume_scaling_windows; do
    cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --ume_rolling_scaling_window $w"
    python3 $cmd
done

# TILTED-CCC
for t in $calibration_targets; do
    cmd="MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC --calibration_target $t"
    python3 $cmd
    for w in $ume_scaling_windows; do
        python3 "$cmd --ume_rolling_scaling_window $w"
    done
done