#!/usr/bin/env bash
#SBATCH --job-name=rez_transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=7-
#SBATCH --cpus=1
#SBATCH --mem-per-cpu=120000

python ./OpenNMT-py/preprocess.py \
-train_src data/src-train.txt \
-train_tgt data/tgt-train.txt \
-valid_src data/src-val.txt \
-valid_tgt data/tgt-val.txt \
-save_data data/demo