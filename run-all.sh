#!/bin/bash

#SBATCH --mem=25000
#SBATCH -J UME

export START_HERE=/nas/student/NicolasKolbenschlag/emotion_uncertainty_bachelorarbeit

source $START_HERE/venv/bin/activate

features="bert-4"
emo_dim="valence"

# GLOBAL
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 1 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 100 --load_subjectivity_from_file"
python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 100"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty_ensemble.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach ensemble_averaging --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --global_uncertainty_window 100 --load_subjectivity_from_file"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty_ensemble.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 2 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach ensemble_averaging --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --measure_uncertainty_globally --load_subjectivity_from_file"
# python3 $cmd

# LOCAL
cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --loss ccc --uncertainty_approach monte_carlo_dropout --attn_dr .5 --out_dr .5 --rnn_n_layers 2 --rnn_dr .5 --load_subjectivity_from_file"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 314 --attn --rnn_bi --uncertainty_approach quantile_regression --loss tiltedCCC --load_subjectivity_from_file"
# python3 $cmd

# PREDICT SUBJECTIVITY
features="vggish"
emo_dim="arousal"

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity ccc --add_seg_id"
# python3 $cmd

cmd="$START_HERE/MuSe-LSTM-Attention-baseline-model/emotion_recognition/main_uncertainty.py --feature_set $features --emo_dim_set $emo_dim --epochs 100 --refresh --n_seeds 1 --seed 43 --attn --rnn_bi --loss ccc --not_measure_uncertainty --predict_subjectivity --loss_subjectivity mse --add_seg_id --load_subjectivity_from_file"
# python3 $cmd