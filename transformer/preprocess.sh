#!/usr/bin/env bash
#SBATCH --job-name=rez_transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=7-
#SBATCH --cpus=1
#SBATCH --mem-per-cpu=120000

python ./OpenNMT-py/preprocess.py \
-train_src data/en.train.enc \
-train_tgt data/et.train.enc \
-valid_src data/en.dev.enc \
-valid_tgt data/et.dev.enc \
-save_data data_v2/demo