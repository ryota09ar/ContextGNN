#!/bin/bash

#SBATCH --job-name=contextgnn_yelp_recent_once
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --error=slurm_contextgnn_yelp_recent_once.err
#SBATCH --output=slurm_contextgnn_yelp_recent_once.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate contextgnn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

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

DATA_DIR="$PWD/data/cache/yelp/transductive_recent_2020-02-01"
SAVE_DIR="$PWD/result/yelp_contextgnn_recent_once"
export MPLCONFIGDIR="$SAVE_DIR/matplotlib"
export XDG_CACHE_HOME="$SAVE_DIR/xdg_cache"

if [ ! -f "$DATA_DIR/metadata.pt" ] || \
   [ ! -f "$DATA_DIR/train.pt" ] || \
   [ ! -f "$DATA_DIR/valid.pt" ] || \
   [ ! -f "$DATA_DIR/test.pt" ]; then
  echo "Yelp dataset is incomplete: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

python examples/yelp_contextgnn.py \
  --data_dir "$DATA_DIR" \
  --lr 0.001 \
  --epochs 20 \
  --batch_size 512 \
  --channels 128 \
  --num_layers 4 \
  --num_neighbors 128 \
  --rhs_sample_size 1000 \
  --train_positive_tower_rate 0.5 \
  --max_steps_per_epoch 2000 \
  --eval_k 20 \
  --filter_train_items \
  --analyze_score_modes \
  --save_dir "$SAVE_DIR"
