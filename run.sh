#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME_random

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 5 --seed 314 --predict --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5
python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 100 --refresh --n_seeds 5 --seed 314 --predict --attn --rnn_bi --loss ccc