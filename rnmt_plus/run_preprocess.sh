#!/bin/bash
#
#SBATCH --job-name=train_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=7-
#SBATCH --cpus=1
#SBATCH --mem-per-cpu=120000

python ./preprocess.py \
-train_src ./en.train.enc \
-train_tgt ./et.train.enc \
-valid_src ./en.dev.enc \
-valid_tgt ./et.dev.enc \
-src_vocab ./en.vocab \
-tgt_vocab ./et.vocab \
-save_data ./data
