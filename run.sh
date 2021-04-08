#!/bin/bash

#SBATCH --mem=20000
#SBATCH -J calibration

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_calibration.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 10 --seed 314 --predict --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .3 --out_dr .3
# python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_calibration.py --feature_set bert-4 --emo_dim_set valence --epochs 1 --refresh --n_seeds 2 --seed 314 --predict --attn --rnn_bi --loss tilted --uncertainty_approach quantile_regression