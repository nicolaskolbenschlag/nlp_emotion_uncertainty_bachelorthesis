#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME_abs

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

# features=bert-4
features=vggish

python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set valence --epochs 100 --refresh --n_seeds 2 --seed 314 --predict --attn --rnn_bi --loss ccc
python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set valence --epochs 100 --refresh --n_seeds 2 --seed 314 --predict --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5

# python3 MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set bert-4 --emo_dim_set valence --epochs 1 --refresh --n_seeds 1 --seed 314 --predict --attn --rnn_bi --loss ccc
