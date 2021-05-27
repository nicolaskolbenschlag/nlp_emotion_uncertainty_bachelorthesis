#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J dropout

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features=bert-4
emo_dim=valence

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 3 --seed 314 --attn --rnn_bi --loss ccc --not_measure_uncertainty --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5"
python3 $cmd