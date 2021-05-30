#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features="bert-4"
emo_dim="valence"

# MONTE CARLO DROPOUT   
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5"
python3 $cmd

# TILTED-CCC
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC"
python3 $cmd

# RANDOM UNCERTAINTY QUANTIFICATIONS
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc"
python3 $cmd

# PREDICT SUBJECTIVITY
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse"
# python3 $cmd