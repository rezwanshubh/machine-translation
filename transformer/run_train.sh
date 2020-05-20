#!/usr/bin/env bash
#SBATCH --job-name=rez_transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=7-
#SBATCH --cpus=1
#SBATCH --mem-per-cpu=120000

python ./OpenNMT-py/train.py -data ./data -save_model ./models/ \
  -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 \
  -heads 8 -encoder_type transformer -decoder_type transformer \
  -position_encoding -train_steps 350000 -dropout 0.1 -batch_size 4096 \
  -gpu_ranks 0

