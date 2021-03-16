#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 48  # number of cores
#SBATCH --gres=gpu:8
#SBATCH -o trn_chck_8_8_trn_b.out # STDOUT
#SBATCH -e trn_chck_8_8_trn_b.err # STDERR
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
python3 ./frd.py transformer transformer_52_2048_4096 8