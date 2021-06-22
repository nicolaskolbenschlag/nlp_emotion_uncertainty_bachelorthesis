#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J dir-subj

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

features="bert-4"
emo_dim="valence"

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc"
python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse"
python3 $cmd

features="vggish"
emo_dim="arousal"

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc --load_subjectivity_from_file"
python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse --load_subjectivity_from_file"
python3 $cmd