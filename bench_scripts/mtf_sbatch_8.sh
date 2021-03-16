#!/bin/bash
#SBATCH -p dgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 48  # number of cores
#SBATCH --gres=gpu:8
#SBATCH -o trn_chck_8_8_mtf.out # STDOUT
#SBATCH -e trn_chck_8_8_mtf.err # STDERR
 rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
srun python3 ./frd.py mtf_transformer mtf_transformer_52_2048_4096_model_8 8 8
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
srun python3 ./frd.py mtf_transformer mtf_transformer_52_2048_4096_model_2_batch_4 8 8
rm -r ./t2t/train/*
rm -r ./t2t/img/train/*
srun python3 ./frd.py mtf_transformer mtf_transformer_52_2048_4096_model_4_batch_2 8 8 