"""Train and evaluate ContextGNN on cached Yelp transductive data."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/contextgnn_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/contextgnn_xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch_frame
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from contextgnn.nn.models import ContextGNN
from contextgnn.utils import RHSEmbeddingMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cache/yelp/transductive_recent_2020-02-01",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_epochs_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--rhs_sample_size", type=int, default=1000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_train_items", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--analyze_score_modes", action="store_true")
    parser.add_argument("--analysis_dir", type=str, default=None)
    parser.add_argument("--analysis_scatter_limit", type=int, default=5000)
    return parser.parse_args()


def make_embedding_tf(feat: Tensor, col_name: str) -> TensorFrame:
    num_rows, dim = feat.size()
    emb = MultiEmbeddingTensor(
        num_rows=num_rows,
        num_cols=1,
        values=feat,
        offset=torch.tensor([0, dim]),
    )
    return TensorFrame(
        feat_dict={torch_frame.embedding: emb},
        col_names_dict={torch_frame.embedding: [col_name]},
        num_rows=num_rows,
    )


def make_positive_index(
    user_ids: Tensor,
    item_ids: Tensor,
    num_users: int,
) -> List[Tensor]:
    items_by_user: List[List[int]] = [[] for _ in range(num_users)]
    for user, item in zip(user_ids.tolist(), item_ids.tolist()):
        items_by_user[user].append(item)
    return [torch.tensor(items, dtype=torch.long) for items in items_by_user]


def clear_rhs_cache(model: ContextGNN) -> None:
    model.rhs_embedding._cached_rhs_embedding = None


def attach_contextgnn_batch_fields(
    batch: HeteroData,
    num_hops: int,
) -> HeteroData:
    batch_size = batch["user"].batch_size
    device = batch["user"].n_id.device
    batch["user"].seed_time = torch.zeros(batch_size, device=device)

    user_batch = torch.zeros(batch["user"].num_nodes, dtype=torch.long)
    user_batch[:batch_size] = torch.arange(batch_size)
    batch["user"].batch = user_batch.to(device)

    item_reach: Dict[int, Set[int]] = {}
    active_users: Dict[int, Set[int]] = {
        local_user: {local_user}
        for local_user in range(batch_size)
    }
    active_items: Dict[int, Set[int]] = {}
    user_to_item_edges = batch["user", "rates", "item"].edge_index.cpu()
    item_to_user_edges = batch["item", "rev_rates", "user"].edge_index.cpu()

    for hop in range(num_hops):
        if hop % 2 == 0:
            next_items: Dict[int, Set[int]] = {}
            for src, dst in zip(
                user_to_item_edges[0].tolist(),
                user_to_item_edges[1].tolist(),
            ):
                rows = active_users.get(src)
                if rows is None:
                    continue
                next_items.setdefault(dst, set()).update(rows)
                item_reach.setdefault(dst, set()).update(rows)
            active_items = next_items
        else:
            next_users: Dict[int, Set[int]] = {}
            for src, dst in zip(
                item_to_user_edges[0].tolist(),
                item_to_user_edges[1].tolist(),
            ):
                rows = active_items.get(src)
                if rows is None:
                    continue
                next_users.setdefault(dst, set()).update(rows)
            active_users = next_users
        if not active_users and not active_items:
            break

    lhs_batches = []
    rhs_local_indices = []
    for local_item_idx, rows in item_reach.items():
        for row in sorted(rows):
            lhs_batches.append(row)
            rhs_local_indices.append(local_item_idx)

    if lhs_batches:
        contextgnn_lhs_batch = torch.tensor(
            lhs_batches, dtype=torch.long, device=device)
        contextgnn_rhs_local_index = torch.tensor(
            rhs_local_indices, dtype=torch.long, device=device)
        contextgnn_rhs_global_index = (
            batch["item"].n_id[contextgnn_rhs_local_index])
    else:
        contextgnn_lhs_batch = torch.empty(0, dtype=torch.long, device=device)
        contextgnn_rhs_local_index = torch.empty(
            0, dtype=torch.long, device=device)
        contextgnn_rhs_global_index = torch.empty(
            0, dtype=torch.long, device=device)

    batch["item"].contextgnn_lhs_batch = contextgnn_lhs_batch
    batch["item"].contextgnn_rhs_local_index = contextgnn_rhs_local_index
    batch["item"].contextgnn_rhs_global_index = contextgnn_rhs_global_index

    # Kept for temporal encoding compatibility. Candidate scoring uses the
    # pair-level fields above because one item can belong to multiple seed rows.
    item_batch = torch.zeros(batch["item"].num_nodes, dtype=torch.long)
    batch["item"].batch = item_batch.to(device)
    return batch


def batch_positives(
    input_user_ids: Tensor,
    positives: List[Tensor],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    src_batches = []
    dst_indices = []
    for batch_idx, user_id in enumerate(input_user_ids.tolist()):
        item_ids = positives[user_id]
        if item_ids.numel() == 0:
            continue
        src_batches.append(
            torch.full((item_ids.numel(),), batch_idx, dtype=torch.long))
        dst_indices.append(item_ids)
    if not src_batches:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return (
        torch.cat(src_batches).to(device),
        torch.cat(dst_indices).to(device),
    )


def build_data(metadata: dict) -> Tuple[HeteroData, Dict[str, dict]]:
    item_feat = metadata["item_feature_table"].float()
    train_user = metadata["train_edges"]["user_ids"].long()
    train_item = metadata["train_edges"]["item_ids"].long()
    num_users = len(metadata["user_index"])
    num_items = len(metadata["business_index"])

    user_feat = torch.zeros(num_users, item_feat.size(1), dtype=item_feat.dtype)
    user_feat.index_add_(0, train_user, item_feat[train_item])
    counts = torch.bincount(train_user, minlength=num_users).clamp_min(1)
    user_feat = user_feat / counts.view(-1, 1)

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["item"].num_nodes = num_items
    data["user"].time = torch.zeros(num_users)
    data["item"].time = torch.zeros(num_items)
    data["user"].tf = make_embedding_tf(user_feat, "user_history_embedding")
    data["item"].tf = make_embedding_tf(item_feat, "item_embedding")
    data["user", "rates", "item"].edge_index = torch.stack(
        [train_user, train_item], dim=0)
    data["item", "rev_rates", "user"].edge_index = torch.stack(
        [train_item, train_user], dim=0)

    dim = int(metadata["item_feature_dim"])
    col_stats_dict = {
        "user": {
            "user_history_embedding": {
                StatType.EMB_DIM: dim,
            },
        },
        "item": {
            "item_embedding": {
                StatType.EMB_DIM: dim,
            },
        },
    }
    return data, col_stats_dict


def make_loader(
    data: HeteroData,
    user_ids: Tensor,
    num_neighbors: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> NeighborLoader:
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("user", user_ids.unique(sorted=True)),
        subgraph_type="bidirectional",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: ContextGNN,
    loader: NeighborLoader,
    positives: List[Tensor],
    train_positives: List[Tensor],
    eval_k: int,
    num_hops: int,
    device: torch.device,
    filter_train_items: bool,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    clear_rhs_cache(model)
    recall_sum = ndcg_sum = map_sum = hit_sum = 0.0
    num_eval_users = 0

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        batch = attach_contextgnn_batch_fields(batch, num_hops)
        user_ids = batch["user"].n_id[:batch["user"].batch_size].cpu()
        scores = model(batch, "user", "item").detach()

        if filter_train_items:
            for row, user_id in enumerate(user_ids.tolist()):
                seen = train_positives[user_id]
                if seen.numel() > 0:
                    scores[row, seen.to(device)] = -float("inf")

        _, pred = torch.topk(scores, k=eval_k, dim=1)
        pred = pred.cpu()

        for row, user_id in enumerate(user_ids.tolist()):
            target = positives[user_id]
            if target.numel() == 0:
                continue
            target_set = set(target.tolist())
            hits = [1 if item in target_set else 0 for item in pred[row].tolist()]
            num_hits = sum(hits)
            if num_hits == 0:
                num_eval_users += 1
                continue

            recall_sum += num_hits / min(len(target_set), eval_k)
            hit_sum += 1.0

            dcg = sum(hit / math.log2(rank + 2)
                      for rank, hit in enumerate(hits))
            ideal_hits = min(len(target_set), eval_k)
            idcg = sum(1.0 / math.log2(rank + 2)
                       for rank in range(ideal_hits))
            ndcg_sum += dcg / idcg

            precision_sum = 0.0
            running_hits = 0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    running_hits += 1
                    precision_sum += running_hits / rank
            map_sum += precision_sum / min(len(target_set), eval_k)
            num_eval_users += 1

    if num_eval_users == 0:
        return {
            f"hit@{eval_k}": 0.0,
            f"recall@{eval_k}": 0.0,
            f"ndcg@{eval_k}": 0.0,
            f"map@{eval_k}": 0.0,
        }
    return {
        f"hit@{eval_k}": hit_sum / num_eval_users,
        f"recall@{eval_k}": recall_sum / num_eval_users,
        f"ndcg@{eval_k}": ndcg_sum / num_eval_users,
        f"map@{eval_k}": map_sum / num_eval_users,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_bar_plot(
    path: Path,
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str,
) -> None:
    fig_width = max(8.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    colors = plt.get_cmap("tab10").colors
    bars = ax.bar(labels, values, color=[colors[i % len(colors)] for i in range(len(labels))])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
    save_figure(fig, path)


def write_grouped_bar_plot(
    path: Path,
    title: str,
    groups: List[str],
    series: Dict[str, List[float]],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = list(range(len(groups)))
    series_items = list(series.items())
    width = min(0.24, 0.8 / max(len(series_items), 1))
    offsets = [
        (idx - (len(series_items) - 1) / 2) * width
        for idx in range(len(series_items))
    ]
    for offset, (name, values) in zip(offsets, series_items):
        positions = [v + offset for v in x]
        bars = ax.bar(positions, values, width=width, label=name)
        ax.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    save_figure(fig, path)


def write_heatmap_plot(path: Path, title: str, labels: List[str],
                       matrix: List[List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f"{matrix[row][col]:.3f}", ha="center",
                    va="center", color="black", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, path)


def write_scatter_plot(path: Path, title: str, points: List[Tuple[float, float]],
                       xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=8, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", alpha=0.35)
    save_figure(fig, path)


def write_hist_plot(path: Path, title: str, values: List[int], xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if values:
        bins = min(41, max(values) - min(values) + 1)
        ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    else:
        ax.hist([0], bins=1, color="#4C78A8", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(fig, path)


def topk_indices(scores: Tensor, k: int) -> List[int]:
    values, indices = torch.topk(scores, k=k)
    finite_mask = torch.isfinite(values)
    return indices[finite_mask].cpu().tolist()


def topk_set(scores: Tensor, k: int) -> set:
    return set(topk_indices(scores, k))


def exact_rank(scores: Tensor, item_id: int) -> int:
    item_score = scores[item_id]
    if torch.isneginf(item_score):
        return -1
    return int((scores > item_score).sum().item()) + 1


@torch.no_grad()
def analyze_score_modes(
    model: ContextGNN,
    loader: NeighborLoader,
    positives: List[Tensor],
    train_positives: List[Tensor],
    eval_k: int,
    num_hops: int,
    device: torch.device,
    filter_train_items: bool,
    output_dir: Path,
    scatter_limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    clear_rhs_cache(model)
    modes = ["fused", "two_tower", "gnn_only"]
    metric_sums: Dict[str, Dict[str, float]] = {
        mode: {"hit": 0.0, "recall": 0.0, "ndcg": 0.0, "map": 0.0}
        for mode in modes
    }
    eval_users = 0
    overlap_sums: Dict[Tuple[str, str], float] = {
        (left, right): 0.0
        for left in modes
        for right in modes
    }
    pair_counts = {
        "lost_from_two_tower": 0,
        "lost_from_gnn": 0,
        "gained_by_fused_vs_two_tower": 0,
        "gained_by_fused_vs_gnn": 0,
        "hit_by_all": 0,
        "missed_by_all": 0,
        "kept_from_two_tower": 0,
        "kept_from_gnn": 0,
    }
    user_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    scatter_points: List[Tuple[float, float]] = []
    rank_shifts: List[int] = []

    for batch in tqdm(loader, desc="Analyze score modes"):
        batch = batch.to(device)
        batch = attach_contextgnn_batch_fields(batch, num_hops)
        user_ids = batch["user"].n_id[:batch["user"].batch_size].cpu()
        components = model.forward_components(batch, "user", "item")
        scores_by_mode = {mode: components[mode].detach().clone() for mode in modes}

        if filter_train_items:
            for row, user_id in enumerate(user_ids.tolist()):
                seen = train_positives[user_id]
                if seen.numel() > 0:
                    seen = seen.to(device)
                    for scores in scores_by_mode.values():
                        scores[row, seen] = -float("inf")

        for row, user_id in enumerate(user_ids.tolist()):
            target = positives[user_id]
            if target.numel() == 0:
                continue
            target_set = set(target.tolist())
            topk_by_mode = {
                mode: topk_set(scores_by_mode[mode][row], eval_k)
                for mode in modes
            }
            eval_users += 1

            user_row: Dict[str, Any] = {
                "user_id": user_id,
                "num_test_items": len(target_set),
            }
            for mode in modes:
                ordered_pred = topk_indices(scores_by_mode[mode][row], eval_k)
                hits = [1 if item in target_set else 0 for item in ordered_pred]
                num_hits = sum(hits)
                metric_sums[mode]["hit"] += 1.0 if num_hits else 0.0
                metric_sums[mode]["recall"] += (
                    num_hits / min(len(target_set), eval_k))
                if num_hits:
                    dcg = sum(hit / math.log2(rank + 2)
                              for rank, hit in enumerate(hits))
                    idcg = sum(1.0 / math.log2(rank + 2)
                               for rank in range(min(len(target_set), eval_k)))
                    metric_sums[mode]["ndcg"] += dcg / idcg
                    precision_sum = 0.0
                    running_hits = 0
                    for rank, hit in enumerate(hits, start=1):
                        if hit:
                            running_hits += 1
                            precision_sum += running_hits / rank
                    metric_sums[mode]["map"] += (
                        precision_sum / min(len(target_set), eval_k))
                user_row[f"hit_{mode}"] = 1 if num_hits else 0
                user_row[f"recall_{mode}"] = (
                    num_hits / min(len(target_set), eval_k))
                user_row[f"topk_positive_count_{mode}"] = num_hits
                user_row[f"topk_{mode}"] = " ".join(map(str, ordered_pred))

            for left in modes:
                for right in modes:
                    union = topk_by_mode[left] | topk_by_mode[right]
                    overlap = 1.0 if not union else (
                        len(topk_by_mode[left] & topk_by_mode[right]) /
                        len(union))
                    overlap_sums[(left, right)] += overlap
                    user_row[f"jaccard_{left}_{right}"] = overlap

            user_row["lost_from_two_tower"] = int(
                bool(topk_by_mode["two_tower"] & target_set)
                and not bool(topk_by_mode["fused"] & target_set))
            user_row["lost_from_gnn"] = int(
                bool(topk_by_mode["gnn_only"] & target_set)
                and not bool(topk_by_mode["fused"] & target_set))
            user_row["gained_by_fused_vs_two_tower"] = int(
                bool(topk_by_mode["fused"] & target_set)
                and not bool(topk_by_mode["two_tower"] & target_set))
            user_row["gained_by_fused_vs_gnn"] = int(
                bool(topk_by_mode["fused"] & target_set)
                and not bool(topk_by_mode["gnn_only"] & target_set))
            user_rows.append(user_row)

            for item_id in target.tolist():
                in_topk = {
                    mode: item_id in topk_by_mode[mode]
                    for mode in modes
                }
                if in_topk["two_tower"] and not in_topk["fused"]:
                    pair_counts["lost_from_two_tower"] += 1
                    case_type = "lost_from_two_tower"
                elif in_topk["gnn_only"] and not in_topk["fused"]:
                    pair_counts["lost_from_gnn"] += 1
                    case_type = "lost_from_gnn"
                elif in_topk["fused"] and not in_topk["two_tower"]:
                    pair_counts["gained_by_fused_vs_two_tower"] += 1
                    case_type = "gained_by_fused_vs_two_tower"
                elif in_topk["fused"] and not in_topk["gnn_only"]:
                    pair_counts["gained_by_fused_vs_gnn"] += 1
                    case_type = "gained_by_fused_vs_gnn"
                elif all(in_topk.values()):
                    pair_counts["hit_by_all"] += 1
                    case_type = "hit_by_all"
                elif not any(in_topk.values()):
                    pair_counts["missed_by_all"] += 1
                    case_type = "missed_by_all"
                elif in_topk["two_tower"] and in_topk["fused"]:
                    pair_counts["kept_from_two_tower"] += 1
                    case_type = "kept_from_two_tower"
                else:
                    pair_counts["kept_from_gnn"] += 1
                    case_type = "kept_from_gnn"

                ranks = {
                    mode: exact_rank(scores_by_mode[mode][row], int(item_id))
                    for mode in modes
                }
                if ranks["two_tower"] > 0 and ranks["fused"] > 0:
                    rank_shifts.append(ranks["two_tower"] - ranks["fused"])
                two_score = float(scores_by_mode["two_tower"][row, item_id])
                fused_score = float(scores_by_mode["fused"][row, item_id])
                gnn_score = float(scores_by_mode["gnn_only"][row, item_id])
                if len(scatter_points) < scatter_limit:
                    scatter_points.append((two_score, fused_score))
                pair_rows.append({
                    "user_id": user_id,
                    "item_id": int(item_id),
                    "case_type": case_type,
                    "in_topk_fused": int(in_topk["fused"]),
                    "in_topk_two_tower": int(in_topk["two_tower"]),
                    "in_topk_gnn_only": int(in_topk["gnn_only"]),
                    "rank_fused": ranks["fused"],
                    "rank_two_tower": ranks["two_tower"],
                    "rank_gnn_only": ranks["gnn_only"],
                    "score_fused": fused_score,
                    "score_two_tower": two_score,
                    "score_gnn_only": gnn_score,
                })

    metrics = {
        mode: {
            f"hit@{eval_k}": metric_sums[mode]["hit"] / eval_users,
            f"recall@{eval_k}": metric_sums[mode]["recall"] / eval_users,
            f"ndcg@{eval_k}": metric_sums[mode]["ndcg"] / eval_users,
            f"map@{eval_k}": metric_sums[mode]["map"] / eval_users,
        }
        for mode in modes
    }
    overlap = {
        left: {
            right: overlap_sums[(left, right)] / eval_users
            for right in modes
        }
        for left in modes
    }
    summary = {
        "eval_k": eval_k,
        "num_eval_users": eval_users,
        "metrics": metrics,
        "metric_delta": {
            "fused_minus_two_tower": {
                key: metrics["fused"][key] - metrics["two_tower"][key]
                for key in metrics["fused"]
            },
            "fused_minus_gnn_only": {
                key: metrics["fused"][key] - metrics["gnn_only"][key]
                for key in metrics["fused"]
            },
        },
        "topk_jaccard": overlap,
        "positive_pair_counts": pair_counts,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(output_dir / "user_level.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(user_rows[0].keys()))
        writer.writeheader()
        writer.writerows(user_rows)
    with open(output_dir / "pair_level.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    metric_names = [f"hit@{eval_k}", f"recall@{eval_k}", f"ndcg@{eval_k}",
                    f"map@{eval_k}"]
    write_grouped_bar_plot(
        output_dir / "metrics_bar.png",
        "Score Mode Metrics",
        metric_names,
        {mode: [metrics[mode][metric] for metric in metric_names]
         for mode in modes},
        "metric value",
    )
    write_grouped_bar_plot(
        output_dir / "delta_bar.png",
        "Fused Metric Delta",
        metric_names,
        {
            "fused - two_tower": [
                summary["metric_delta"]["fused_minus_two_tower"][metric]
                for metric in metric_names
            ],
            "fused - gnn_only": [
                summary["metric_delta"]["fused_minus_gnn_only"][metric]
                for metric in metric_names
            ],
        },
        "delta",
    )
    write_heatmap_plot(
        output_dir / "topk_overlap_heatmap.png",
        "Top-K Jaccard Overlap",
        modes,
        [[overlap[left][right] for right in modes] for left in modes],
    )
    write_bar_plot(
        output_dir / "lost_gained_counts.png",
        "Positive Pair Lost/Gained Counts",
        list(pair_counts.keys()),
        [float(value) for value in pair_counts.values()],
        "positive pairs",
    )
    write_scatter_plot(
        output_dir / "score_scatter_fused_vs_two_tower.png",
        "Positive Pair Scores",
        scatter_points,
        "two_tower score",
        "fused score",
    )
    write_hist_plot(
        output_dir / "rank_shift_hist.png",
        "Positive Pair Rank Shift: two_tower rank - fused rank",
        rank_shifts,
        "rank shift",
    )

    md_lines = [
        "# ContextGNN Score Mode Analysis",
        "",
        f"- eval_k: {eval_k}",
        f"- num_eval_users: {eval_users}",
        "",
        "## Metrics",
        "",
        "| mode | hit | recall | ndcg | map |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        md_lines.append(
            f"| {mode} | {metrics[mode][f'hit@{eval_k}']:.8f} | "
            f"{metrics[mode][f'recall@{eval_k}']:.8f} | "
            f"{metrics[mode][f'ndcg@{eval_k}']:.8f} | "
            f"{metrics[mode][f'map@{eval_k}']:.8f} |")
    md_lines.extend([
        "",
        "## Delta",
        "",
        "| comparison | hit | recall | ndcg | map |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, deltas in summary["metric_delta"].items():
        md_lines.append(
            f"| {name} | {deltas[f'hit@{eval_k}']:.8f} | "
            f"{deltas[f'recall@{eval_k}']:.8f} | "
            f"{deltas[f'ndcg@{eval_k}']:.8f} | "
            f"{deltas[f'map@{eval_k}']:.8f} |")
    md_lines.extend([
        "",
        "## Positive Pair Counts",
        "",
        "| case | count |",
        "| --- | ---: |",
    ])
    for key, value in pair_counts.items():
        md_lines.append(f"| {key} | {value} |")
    md_lines.extend([
        "",
        "## Files",
        "",
        "- `summary.json`",
        "- `user_level.csv`",
        "- `pair_level.csv`",
        "- `metrics_bar.png` / `metrics_bar.svg`",
        "- `delta_bar.png` / `delta_bar.svg`",
        "- `topk_overlap_heatmap.png` / `topk_overlap_heatmap.svg`",
        "- `lost_gained_counts.png` / `lost_gained_counts.svg`",
        "- `score_scatter_fused_vs_two_tower.png` / `score_scatter_fused_vs_two_tower.svg`",
        "- `rank_shift_hist.png` / `rank_shift_hist.svg`",
    ])
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    metadata = torch.load(data_dir / "metadata.pt", map_location="cpu")
    split_edges = {
        split: torch.load(data_dir / f"{split}.pt", map_location="cpu")
        for split in ["train", "valid", "test"]
    }

    data, col_stats_dict = build_data(metadata)
    num_neighbors = [
        int(args.num_neighbors // 2**i) for i in range(args.num_layers)
    ]
    loaders = {
        split: make_loader(
            data=data,
            user_ids=edges["user_ids"].long(),
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
        )
        for split, edges in split_edges.items()
    }

    positives = {
        split: make_positive_index(
            edges["user_ids"].long(),
            edges["item_ids"].long(),
            data["user"].num_nodes,
        )
        for split, edges in split_edges.items()
    }

    model = ContextGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table="item",
        num_nodes=data["item"].num_nodes,
        num_layers=args.num_layers,
        channels=args.channels,
        aggr=args.aggr,
        norm="layer_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": 4,
        },
        rhs_sample_size=args.rhs_sample_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state_dict = None
    best_valid_metric = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_count = steps = 0
        total_steps = min(len(loaders["train"]), args.max_steps_per_epoch)

        for batch in tqdm(loaders["train"], total=total_steps, desc="Train"):
            batch = batch.to(device)
            batch = attach_contextgnn_batch_fields(batch, args.num_layers)
            input_user_ids = batch["user"].n_id[:batch["user"].batch_size].cpu()
            src_batch, dst_index = batch_positives(
                input_user_ids,
                positives["train"],
                device,
            )
            if dst_index.numel() == 0:
                continue

            optimizer.zero_grad()
            logits, lhs_y_batch, rhs_y_index = model.forward_sample_softmax(
                batch, "user", "item", src_batch, dst_index)
            edge_label_index = torch.stack([lhs_y_batch, rhs_y_index], dim=0)
            loss = sparse_cross_entropy(logits, edge_label_index)
            loss.backward()
            optimizer.step()

            count = int(batch["user"].batch_size)
            total_loss += float(loss.detach()) * count
            total_count += count
            steps += 1
            if steps >= args.max_steps_per_epoch:
                break

        train_loss = total_loss / total_count if total_count else float("nan")
        if epoch % args.eval_epochs_interval == 0:
            valid_metrics = evaluate(
                model=model,
                loader=loaders["valid"],
                positives=positives["valid"],
                train_positives=positives["train"],
                eval_k=args.eval_k,
                num_hops=args.num_layers,
                device=device,
                filter_train_items=args.filter_train_items,
                desc="Valid",
            )
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
                  f"Valid metrics: {valid_metrics}")

            valid_metric = valid_metrics[f"map@{args.eval_k}"]
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_state_dict = {
                    key: value.cpu()
                    for key, value in model.state_dict().items()
                }

    if best_state_dict is None:
        raise RuntimeError("No validation checkpoint was selected.")

    model.load_state_dict(best_state_dict)
    clear_rhs_cache(model)
    model = model.to(device)
    valid_metrics = evaluate(
        model=model,
        loader=loaders["valid"],
        positives=positives["valid"],
        train_positives=positives["train"],
        eval_k=args.eval_k,
        num_hops=args.num_layers,
        device=device,
        filter_train_items=args.filter_train_items,
        desc="Best valid",
    )
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        positives=positives["test"],
        train_positives=positives["train"],
        eval_k=args.eval_k,
        num_hops=args.num_layers,
        device=device,
        filter_train_items=args.filter_train_items,
        desc="Test",
    )
    print(f"Best valid metrics: {valid_metrics}")
    print(f"Best test metrics: {test_metrics}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, save_dir / "contextgnn_yelp_best.pt")
        with open(save_dir / "metrics.json", "w") as f:
            json.dump({
                "best_valid_metrics": valid_metrics,
                "best_test_metrics": test_metrics,
                "args": vars(args),
            }, f, indent=2)

    if args.analyze_score_modes:
        if args.analysis_dir is not None:
            analysis_dir = Path(args.analysis_dir)
        elif args.save_dir is not None:
            analysis_dir = Path(args.save_dir) / "analysis"
        else:
            analysis_dir = Path("result/yelp_contextgnn_analysis")
        analyze_score_modes(
            model=model,
            loader=loaders["test"],
            positives=positives["test"],
            train_positives=positives["train"],
            eval_k=args.eval_k,
            num_hops=args.num_layers,
            device=device,
            filter_train_items=args.filter_train_items,
            output_dir=analysis_dir,
            scatter_limit=args.analysis_scatter_limit,
        )
        print(f"Score mode analysis saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
