## ContextGNN

このリポジトリには、ContextGNN の PyTorch 実装が含まれています。

ContextGNN は、RelBench などのリレーショナルデータをグラフとして扱い、リンク予測・推薦タスクを学習するためのモデルです。初めてこのディレクトリを触る場合は、まず RelBench の小さいデータセットで動作確認するのがおすすめです。

## できること

- RelBench のリンク予測タスクで ContextGNN を学習する
- 学習後に validation/test データへ推論し、評価指標を表示する
- Optuna を使って RelBench / IJCAI-Contest のベンチマークを再現する
- RHS（右辺、推薦候補側）ノードをサンプリングしてメモリ使用量を抑える

## 前提環境

- Python 3.8 以上
- conda / Miniconda
- PyTorch
- PyTorch Geometric
- pytorch-frame
- examples / benchmarks を実行する場合は RelBench、sentence-transformers、Optuna

GPU が使える環境では CUDA 版 PyTorch を入れると高速に実行できます。CUDA の有無にかかわらず、スクリプト側では `torch.cuda.is_available()` に応じて GPU / CPU を自動選択します。

## インストール

この README では、Python の環境管理に conda を使います。`scripts/script.sh` と `scripts/relbench_trial.sbatch` は、デフォルトで `contextgnn` という conda 環境を有効化して実行します。

```sh
conda create -n contextgnn python=3.11
conda activate contextgnn
```

次に PyTorch をインストールします。CUDA を使う場合は、自分の CUDA バージョンに合う PyTorch を入れてください。CPU のみで試す場合の例は以下です。

```sh
pip install torch
```

このリポジトリ本体と、examples / benchmarks 用の追加依存関係をインストールします。

```sh
pip install -e '.[full]'
```

`.[full]` には主に以下が含まれます。

- `relbench==1.1.0`
- `sentence-transformers`
- `optuna`

別名の conda 環境を使う場合は、`scripts/script.sh` の `conda activate contextgnn` を書き換えるか、`scripts/relbench_trial.sbatch` では `CONDA_ENV_NAME` を指定して実行してください。

```sh
CONDA_ENV_NAME=my-env sbatch scripts/relbench_trial.sbatch
```

## 最初の動作確認

まずは RelBench の `rel-trial` データセットで 1 epoch だけ実行します。初回実行時はデータセットや前処理結果がダウンロード・キャッシュされるため、時間がかかることがあります。

```sh
python examples/relbench_example.py \
  --dataset rel-trial \
  --task site-sponsor-run \
  --model contextgnn \
  --epochs 1 \
  --max_steps_per_epoch 10
```

正常に進むと、学習の進捗バーが表示され、最後に validation と test の評価結果が出力されます。

Slurm 環境で `sbatch` から実行する場合は、先にログインノードで conda 環境と依存関係のインストールを済ませてから、以下を実行してください。

```sh
sbatch scripts/relbench_trial.sbatch
```

このジョブは `scripts/relbench_trial.sbatch` に定義されています。ジョブ内では `~/miniconda3/etc/profile.d/conda.sh` を読み込み、デフォルトで `contextgnn` 環境を有効化します。標準出力は `slurm_contextgnn_relbench_trial.out`、標準エラーは `slurm_contextgnn_relbench_trial.err` に出力されます。

ジョブのキャッシュは XDG の慣例に合わせて、デフォルトでは以下に保存されます。

- RelBench / 前処理キャッシュ: `~/.cache/contextgnn/relbench_examples`
- Hugging Face キャッシュ: `~/.cache/huggingface`
- sentence-transformers キャッシュ: `~/.cache/torch/sentence_transformers`

別の保存先を使う場合は、`XDG_CACHE_HOME` や `CACHE_DIR` を指定して実行してください。

```sh
XDG_CACHE_HOME=/path/to/cache sbatch scripts/relbench_trial.sbatch
```

## RelBench で学習する

基本的な学習コマンドは以下です。

```sh
python examples/relbench_example.py \
  --dataset rel-trial \
  --task site-sponsor-run \
  --model contextgnn
```

主な引数は以下です。

- `--dataset`: RelBench のデータセット名。例: `rel-trial`
- `--task`: RelBench のタスク名。例: `site-sponsor-run`
- `--model`: 使用するモデル。`contextgnn`, `idgnn`, `shallowrhsgnn` から選択
- `--epochs`: 学習 epoch 数。デフォルトは `20`
- `--batch_size`: バッチサイズ。GPU メモリ不足時は小さくする
- `--num_layers`: GNN の層数
- `--num_neighbors`: NeighborLoader でサンプリングする近傍数
- `--max_steps_per_epoch`: 1 epoch あたりの最大ステップ数
- `--cache_dir`: RelBench データと前処理結果のキャッシュ先。デフォルトは `~/.cache/relbench_examples`

GPU メモリが足りない場合は、まず以下を小さくしてください。

```sh
python examples/relbench_example.py \
  --dataset rel-trial \
  --task site-sponsor-run \
  --model contextgnn \
  --batch_size 128 \
  --num_neighbors 32 \
  --max_steps_per_epoch 100
```

## 推論と評価

現在の example スクリプトは、学習、validation 推論、test 推論、評価を 1 回の実行内で行います。つまり、上の学習コマンドを実行すると、学習後に自動で推論と評価まで実行されます。

`examples/relbench_example.py` では、各 epoch の validation 評価が表示され、最も良い validation スコアのモデル重みをメモリ上で保持します。学習後、その重みを使って validation と test に対する推論を行い、評価指標を表示します。

注意点として、現状の example スクリプトは学習済みモデルの checkpoint をファイルには保存しません。別プロセスで後から推論したい場合は、`torch.save(model.state_dict(), ...)` と `model.load_state_dict(...)` を追加してください。

## RHS サンプリングを使う

推薦候補側のノード数が多い場合は、RHS ノードをサンプリングする実装を使えます。

```sh
python examples/contextgnn_sample_softmax.py \
  --dataset rel-amazon \
  --task user-item-purchase \
  --rhs_sample_size 1000
```

`--rhs_sample_size -1` を指定すると、RHS をサンプリングせず全候補を使います。メモリ使用量が増えるため、大きなデータセットでは注意してください。

## ベンチマークを再現する

RelBench のリンク予測ベンチマークを実行するには、以下を使います。

```sh
python benchmark/relbench_link_prediction_benchmark.py \
  --dataset rel-amazon \
  --task user-item-rate \
  --model contextgnn
```

このスクリプトは Optuna によるハイパーパラメータ探索を行います。短時間で動作確認したい場合は、`--num_trials` と `--num_repeats` を小さくしてください。

```sh
python benchmark/relbench_link_prediction_benchmark.py \
  --dataset rel-trial \
  --task site-sponsor-run \
  --model contextgnn \
  --epochs 1 \
  --num_trials 1 \
  --num_repeats 1 \
  --max_steps_per_epoch 10
```

結果はデフォルトで `result/` に保存されます。

## IJCAI-Contest を実行する

IJCAI-Contest 用のスクリプトは、`.data/ijcai-contest` にデータが配置されている前提です。コード上では以下のファイルを読みます。

- `.data/ijcai-contest/trn_click`
- `.data/ijcai-contest/trn_fav`
- `.data/ijcai-contest/trn_cart`
- `.data/ijcai-contest/trn_buy`
- `.data/ijcai-contest/tst_int`

データを配置した後、example は以下で実行できます。

```sh
python examples/ijcai_example.py --model contextgnn
```

IJCAI-Contest のベンチマークを実行するには、以下を使います。

```sh
python benchmark/tgt_ijcai_benchmark.py --model contextgnn
```

短時間で動作確認したい場合は、探索回数と epoch 数を小さくします。

```sh
python benchmark/tgt_ijcai_benchmark.py \
  --model contextgnn \
  --epochs 1 \
  --num_trials 1 \
  --num_repeats 1 \
  --max_steps_per_epoch 10
```

結果はデフォルトで `results/` に保存されます。

## よく使うコマンドまとめ

RelBench の最小動作確認:

```sh
python examples/relbench_example.py --dataset rel-trial --task site-sponsor-run --model contextgnn --epochs 1 --max_steps_per_epoch 10
```

RelBench の通常実行:

```sh
python examples/relbench_example.py --dataset rel-trial --task site-sponsor-run --model contextgnn
```

RHS サンプリング:

```sh
python examples/contextgnn_sample_softmax.py --dataset rel-amazon --task user-item-purchase --rhs_sample_size 1000
```

RelBench ベンチマーク:

```sh
python benchmark/relbench_link_prediction_benchmark.py --model contextgnn
```

IJCAI-Contest ベンチマーク:

```sh
python benchmark/tgt_ijcai_benchmark.py --model contextgnn
```
