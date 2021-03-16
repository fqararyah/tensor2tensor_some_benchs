#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 32  # number of cores
#SBATCH --gres=gpu:4
#SBATCH -o trn_chck_4_8_mtf.out # STDOUT
#SBATCH -e trn_chck_4_8_mtf.err # STDERR
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
srun python3 ./frd.py mtf_transformer mtf_transformer_52_2048_4096_model_4 8 4
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
srun python3 ./frd.py mtf_transformer mtf_transformer_52_2048_4096_model_2_batch_2 8 4