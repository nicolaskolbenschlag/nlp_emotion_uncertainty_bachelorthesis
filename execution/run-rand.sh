#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features="bert-4"
emo_dim="valence"

ume_scaling_windows="10 200 500"

# RANDOM UNCERTAINTY QUANTIFICATIONS
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc"
python3 $cmd

for w in $ume_scaling_windows; do
    python3 "$cmd --ume_rolling_scaling_window $w"
done