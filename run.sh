#!/bin/bash

#SBATCH --mem=20000
#SBATCH -J ccc

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate
# python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 10 --seed 314 --predict --attn --rnn_bi --loss tiltedCCC --uncertainty_approach quantile_regression
python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 10 --seed 314 --predict --attn --rnn_bi --loss ccc