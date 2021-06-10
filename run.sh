#!/bin/bash

#SBATCH --mem=30000
#SBATCH -J gENCE

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

features=bert-4
emo_dim=valence

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty_ensemble.py --feature_set $features --emo_dim_set $emo_dim --epochs 1 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach ensemble_averaging --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20 --normalize_uncalibrated_global_uncertainty_measurement --save_subjectivity_to_file"
python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 1 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 20 --normalize_uncalibrated_global_uncertainty_measurement --load_subjectivity_from_file"
python3 $cmd