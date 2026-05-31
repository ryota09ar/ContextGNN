#!/bin/bash

#SBATCH --job-name=NCGR              # Job name

#SBATCH --mem=8G                       # Job memory request

#SBATCH --gres=gpu:1080ti:1             # Number of requested GPU(s)

#SBATCH --time=10:00:00                   # Time limit days-hrs:min:sec

#SBATCH --error=slurm.err                # Error file name

#SBATCH --output=slurm.out               # Output file name

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contextgnn

python train.py --dataset Pet_Supplies

python evaluate.py \
    --checkpoint checkpoints/Pet_Supplies/best_model_run1.pt \
    --dataset Pet_Supplies \
    --split test
