#!/bin/bash

#SBATCH --job-name=relbench_subset_2017
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --error=slurm_relbench_subset_2017.err
#SBATCH --output=slurm_relbench_subset_2017.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contextgnn

python scripts/create_relbench_time_subset.py \
  --cutoff 2017-01-28 \
  --task_keep_ratio 0.2 \
  --source data/cache/relbench/rel-amazon \
  --output data/cache/relbench/rel-amazon_since_2017-01-28 \
  --overwrite
