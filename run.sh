#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features="bert-4 vggish"
features=bert-4
emo_dim=valence

cmd_base="MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --predict --attn --rnn_bi --loss ccc"
cmd_mcd="$cmd_base --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5"

python3 $cmd_base
python3 $cmd_mcd