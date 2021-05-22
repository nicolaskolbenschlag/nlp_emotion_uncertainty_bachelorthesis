#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J subj

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features=bert-4
emo_dim=valence

# cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --ume_rolling_scaling_window 200"

# cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 1 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity"
python3 $cmd