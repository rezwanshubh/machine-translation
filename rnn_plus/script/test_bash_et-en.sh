#!/bin/bash
#
#SBATCH --job-name=rez_preprocessed_et-en
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=7-
#SBATCH --cpus=1
#SBATCH --mem-per-cpu=12000
#SBATCH --gres=gpu:tesla:1
module unload cudnn-6.0
module load cudnn-5.1

python ./rnn-plus/preprocess.py -train_src ../dataset/et.train.enc -train_tgt ../dataset/en.train.enc -valid_src ../dataset/et.dev.enc -valid_tgt ../dataset/en.dev.enc -src_vocab ../dataset/et.train.enc.json -tgt_vocab ../dataset/en.train.enc.json -save_data ./saved_preprocessed_data_15.pt -max_len 1000