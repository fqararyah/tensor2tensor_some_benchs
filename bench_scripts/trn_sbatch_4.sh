#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 32  # number of cores
#SBATCH --gres=gpu:4
#SBATCH -o trn_chck_4_4_trn_b.out # STDOUT
#SBATCH -e trn_chck_4_4_trn_b.err # STDERR
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
python3 ./frd.py transformer transformer_52_2048_4096 4