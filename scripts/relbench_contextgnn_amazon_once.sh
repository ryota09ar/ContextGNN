#!/bin/bash

#SBATCH --job-name=contextgnn_relbench_amazon_once
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=20:00:00
#SBATCH --error=slurm_contextgnn_relbench_amazon_once.err
#SBATCH --output=slurm_contextgnn_relbench_amazon_once.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contextgnn

ensure_pyg_neighbor_sampler_backend() {
  python - <<'PY'
import importlib.util

has_backend = (
    importlib.util.find_spec("pyg_lib") is not None
    or importlib.util.find_spec("torch_sparse") is not None
)
raise SystemExit(0 if has_backend else 1)
PY
}

if ! ensure_pyg_neighbor_sampler_backend; then
  torch_version=$(python - <<'PY'
import torch

print(torch.__version__.split("+", 1)[0])
PY
)
  cuda_tag=$(python - <<'PY'
import torch

cuda = torch.version.cuda
print("cpu" if cuda is None else "cu" + cuda.replace(".", ""))
PY
)

  if [[ "$torch_version" == 2.12.* ]]; then
    if [[ "$cuda_tag" == "cpu" ]]; then
      torch_index_url="https://download.pytorch.org/whl/cpu"
    else
      torch_index_url="https://download.pytorch.org/whl/$cuda_tag"
    fi

    echo "PyG does not provide prebuilt pyg_lib wheels for torch $torch_version+$cuda_tag; installing torch 2.11.0+$cuda_tag first." >&2
    python -m pip install --upgrade --no-cache-dir \
      "torch==2.11.0" \
      --index-url "$torch_index_url" || exit 1
    torch_version="2.11.0"
  fi

  case "$torch_version" in
    2.11.*) pyg_torch_version="2.11.0" ;;
    2.10.*) pyg_torch_version="2.10.0" ;;
    2.9.*) pyg_torch_version="2.9.0" ;;
    2.8.*) pyg_torch_version="2.8.0" ;;
    2.7.*) pyg_torch_version="2.7.0" ;;
    2.6.*) pyg_torch_version="2.6.0" ;;
    2.5.*) pyg_torch_version="2.5.0" ;;
    2.4.*) pyg_torch_version="2.4.0" ;;
    2.3.*) pyg_torch_version="2.3.0" ;;
    2.2.*) pyg_torch_version="2.2.0" ;;
    2.1.*) pyg_torch_version="2.1.0" ;;
    2.0.*) pyg_torch_version="2.0.0" ;;
    1.13.*) pyg_torch_version="1.13.0" ;;
    *)
      echo "Unsupported PyTorch version for PyG prebuilt pyg_lib wheels: $torch_version+$cuda_tag" >&2
      exit 1
      ;;
  esac

  python -m pip install --upgrade --no-cache-dir \
    pyg_lib \
    -f "https://data.pyg.org/whl/torch-${pyg_torch_version}+${cuda_tag}.html" || exit 1
fi

if ! ensure_pyg_neighbor_sampler_backend; then
  echo "Failed to install pyg_lib or torch_sparse; NeighborSampler cannot run." >&2
  exit 1
fi

SUBSET_DIR="$PWD/data/cache/relbench/rel-amazon_since_2017-01-28"
RUN_CACHE_HOME="$PWD/data/cache_subset_2017"

if [ ! -f "$SUBSET_DIR/db/review.parquet" ] || \
   [ ! -f "$SUBSET_DIR/db/customer.parquet" ] || \
   [ ! -f "$SUBSET_DIR/db/product.parquet" ] || \
   [ ! -f "$SUBSET_DIR/tasks/user-item-rate/train.parquet" ] || \
   [ ! -f "$SUBSET_DIR/tasks/user-item-rate/val.parquet" ] || \
   [ ! -f "$SUBSET_DIR/tasks/user-item-rate/test.parquet" ]; then
  echo "Subset dataset is incomplete: $SUBSET_DIR" >&2
  exit 1
fi

export XDG_CACHE_HOME="$RUN_CACHE_HOME"
mkdir -p "$XDG_CACHE_HOME" "$PWD/data/relbench_examples_subset_2017"

python examples/relbench_example.py \
  --dataset rel-amazon \
  --task user-item-rate \
  --model contextgnn \
  --relbench_cache_dir "$SUBSET_DIR" \
  --lr 0.001 \
  --epochs 20 \
  --batch_size 512 \
  --channels 128 \
  --num_layers 4 \
  --num_neighbors 128 \
  --rhs_sample_size 1000 \
  --max_steps_per_epoch 2000 \
  --cache_dir data/relbench_examples_subset_2017
