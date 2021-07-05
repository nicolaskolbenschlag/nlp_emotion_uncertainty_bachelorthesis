#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J uncertainty

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

features="bert-4"
emo_dim="valence"

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC"
python3 $cmd