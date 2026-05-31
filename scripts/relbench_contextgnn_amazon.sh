#!/bin/bash

#SBATCH --job-name=contextgnn_relbench_amazon
#SBATCH --mem=8G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=10:00:00
#SBATCH --error=slurm_contextgnn_relbench_amazon.err
#SBATCH --output=slurm_contextgnn_relbench_amazon.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contextgnn

export XDG_CACHE_HOME="$PWD/data/cache"
mkdir -p "$XDG_CACHE_HOME" "$PWD/data/relbench_examples"

python benchmark/relbench_link_prediction_benchmark.py \
  --dataset rel-amazon \
  --task user-item-rate \
  --model contextgnn \
  --cache_dir data/relbench_examples
